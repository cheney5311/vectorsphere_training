"""数据集详细管理API接口

提供数据集详细管理相关的API接口，包括数据集的获取、更新、删除、下载、预览、分析和分割功能。
所有接口都需要JWT认证。

API端点:
    - GET /<dataset_id>: 获取数据集详情
    - PUT /<dataset_id>: 更新数据集
    - DELETE /<dataset_id>: 删除数据集
    - GET /<dataset_id>/download: 下载数据集
    - GET /<dataset_id>/preview: 预览数据集
    - POST /<dataset_id>/analyze: 分析数据集
    - POST /<dataset_id>/split: 分割数据集
    - GET /<dataset_id>/statistics: 获取数据集统计信息
    - POST /<dataset_id>/tags: 添加标签
    - GET /<dataset_id>/tags: 获取标签列表
    - DELETE /<dataset_id>/tags/<tag>: 删除标签
    - GET /<dataset_id>/versions: 获取版本历史
    - POST /<dataset_id>/versions: 创建新版本
"""

import sys
import os
from flask import Blueprint, request, jsonify, send_file
from flask_jwt_extended import jwt_required, get_jwt_identity
from werkzeug.utils import secure_filename
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional
import logging

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from backend.core.exceptions import ValidationError
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
dataset_detailed_bp = Blueprint('dataset_detailed', __name__, url_prefix='/api/v1/dataset-detailed')

# 数据集存储目录
DATASET_BASE_DIR = './data/datasets'

# 支持的文件类型
ALLOWED_EXTENSIONS = {'.json', '.jsonl', '.txt', '.csv', '.parquet'}

# 初始化服务
dataset_repository = DatasetRepository()
dataset_service = DatasetService(dataset_repository)


def allowed_file(filename: str) -> bool:
    """检查文件类型是否允许
    
    Args:
        filename: 文件名
        
    Returns:
        bool: 文件类型是否允许上传
    """
    return '.' in filename and \
        os.path.splitext(filename.lower())[1] in ALLOWED_EXTENSIONS


# ============================================================================
# 数据集基本操作API
# ============================================================================

@dataset_detailed_bp.route('/<dataset_id>', methods=['GET'])
@jwt_required()
def get_dataset(dataset_id: str):
    """获取数据集详情
    
    根据数据集ID获取完整的数据集详情信息，包括基本信息、统计信息和标签。
    需要JWT认证，且只能获取当前用户拥有的数据集。
    
    路径参数:
        dataset_id (str): 数据集唯一标识符 (UUID格式)
            - 格式: "550e8400-e29b-41d4-a716-446655440000"
            - 必填
    
    请求头:
        Authorization (str): Bearer <access_token>
            - 格式: "Bearer eyJhbGciOiJIUzI1NiIs..."
            - 必填
    
    响应体 (JSON):
        成功 (200):
            {
                "code": 200,
                "message": "获取数据集详情成功",
                "data": {
                    "dataset": {
                        "dataset_id": "string",      // 数据集唯一ID
                        "user_id": "string",         // 所属用户ID
                        "name": "string",            // 数据集名称
                        "description": "string",     // 数据集描述
                        "dataset_type": "string",    // 数据集类型 (text/image/audio/video/tabular/mixed)
                        "format": "string",          // 数据格式 (json/csv/parquet等)
                        "status": "string",          // 状态 (pending/processing/ready/error/archived)
                        "size": "integer",           // 数据集大小(字节)
                        "record_count": "integer",   // 记录数
                        "storage_path": "string",    // 存储路径
                        "version": "string",         // 版本号
                        "checksum": "string",        // 校验和
                        "ready": "boolean",          // 是否就绪
                        "validated": "boolean",      // 是否已验证
                        "created_at": "string",      // 创建时间 (ISO8601)
                        "updated_at": "string",      // 更新时间 (ISO8601)
                        "config": {},                // 配置信息
                        "features": {},              // 特征信息
                        "labels": {}                 // 标签信息
                    },
                    "tags": ["string"],              // 数据集标签列表
                    "statistics": {                  // 统计信息 (如果存在)
                        "total_rows": "integer",
                        "total_columns": "integer",
                        "column_stats": []
                    }
                }
            }
        失败:
            - 400: 请求参数错误
            - 401: 未授权
            - 403: 无权访问此数据集
            - 404: 数据集不存在
            - 500: 服务器内部错误
    
    示例:
        curl -X GET "http://localhost:5000/api/v1/dataset-detailed/550e8400-e29b-41d4-a716-446655440000" \\
             -H "Authorization: Bearer <token>"
    """
    try:
        user_id = get_jwt_identity()
        
        # 获取数据集
        dataset = dataset_service.get_dataset(dataset_id)
        
        if not dataset:
            return error_response(
                message=f"数据集 {dataset_id} 不存在",
                code=404,
                error_type="DATASET_NOT_FOUND"
            ), 404
        
        # 验证权限
        if dataset.user_id != user_id:
            return error_response(
                message="无权访问此数据集",
                code=403,
                error_type="FORBIDDEN"
            ), 403
        
        # 获取标签
        tags = dataset_service.get_dataset_tags(dataset_id)
        
        # 获取统计信息（如果有）
        statistics = dataset_service.get_dataset_statistics(dataset_id)
        
        # 记录访问日志
        dataset_service.log_access(dataset_id, user_id, "view")
        
        response_data = {
            "dataset": dataset.to_dict(),
            "tags": tags
        }
        
        if statistics:
            response_data["statistics"] = statistics.to_dict() if hasattr(statistics, 'to_dict') else statistics
        
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
    except DatasetValidationError as e:
        return error_response(
            message=str(e),
            code=400,
            error_type="VALIDATION_ERROR"
        ), 400
    except Exception as e:
        logger.error(f"Error getting dataset {dataset_id}: {e}")
        return error_response(
            message=f"获取数据集详情失败: {str(e)}",
            code=500,
            error_type="GET_DATASET_ERROR"
        ), 500


@dataset_detailed_bp.route('/<dataset_id>', methods=['PUT'])
@jwt_required()
def update_dataset(dataset_id: str):
    """更新数据集
    
    更新指定数据集的基本信息，包括名称、描述、配置等。
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
            "name": "string",           // 数据集名称 (可选，1-200字符)
            "description": "string",    // 数据集描述 (可选，最大5000字符)
            "config": {},               // 配置信息 (可选)
            "tags": ["string"]          // 标签列表 (可选)
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
                        "description": "string",
                        "updated_at": "string"
                        // ...其他字段
                    }
                }
            }
        失败:
            - 400: 请求数据无效或验证失败
            - 401: 未授权
            - 403: 无权访问此数据集
            - 404: 数据集不存在
            - 500: 服务器内部错误
    
    示例:
        curl -X PUT "http://localhost:5000/api/v1/dataset-detailed/550e8400-e29b-41d4-a716-446655440000" \\
             -H "Authorization: Bearer <token>" \\
             -H "Content-Type: application/json" \\
             -d '{"name": "新名称", "description": "新描述"}'
    """
    try:
        user_id = get_jwt_identity()
        data = request.get_json()
        
        if not data:
            return error_response(
                message="请求数据不能为空",
                code=400,
                error_type="VALIDATION_ERROR"
            ), 400
        
        # 获取数据集验证权限
        dataset = dataset_service.get_dataset(dataset_id)
        if not dataset:
            return error_response(
                message=f"数据集 {dataset_id} 不存在",
                code=404,
                error_type="DATASET_NOT_FOUND"
            ), 404
        
        if dataset.user_id != user_id:
            return error_response(
                message="无权修改此数据集",
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
        
        # 更新标签（如果提供）
        if 'tags' in data:
            tags = data['tags']
            if isinstance(tags, list):
                # 清除旧标签，添加新标签
                dataset_service.clear_dataset_tags(dataset_id)
                for tag in tags:
                    dataset_service.add_dataset_tag(dataset_id, tag, user_id)
        
        # 记录访问日志
        dataset_service.log_access(dataset_id, user_id, "update")
        
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
            error_type="UPDATE_DATASET_ERROR"
        ), 500


@dataset_detailed_bp.route('/<dataset_id>', methods=['DELETE'])
@jwt_required()
def delete_dataset(dataset_id: str):
    """删除数据集
    
    永久删除指定的数据集，包括数据库记录和存储的文件。
    此操作不可逆，请谨慎使用。
    需要JWT认证，且只能删除当前用户拥有的数据集。
    
    路径参数:
        dataset_id (str): 数据集唯一标识符 (UUID格式)
            - 必填
    
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
            - 401: 未授权
            - 403: 无权删除此数据集
            - 404: 数据集不存在
            - 500: 服务器内部错误
    
    示例:
        curl -X DELETE "http://localhost:5000/api/v1/dataset-detailed/550e8400-e29b-41d4-a716-446655440000" \\
             -H "Authorization: Bearer <token>"
    """
    try:
        user_id = get_jwt_identity()
        
        # 获取数据集验证权限
        dataset = dataset_service.get_dataset(dataset_id)
        if not dataset:
            return error_response(
                message=f"数据集 {dataset_id} 不存在",
                code=404,
                error_type="DATASET_NOT_FOUND"
            ), 404
        
        if dataset.user_id != user_id:
            return error_response(
                message="无权删除此数据集",
                code=403,
                error_type="FORBIDDEN"
            ), 403
        
        # 删除数据集
        success = dataset_service.delete_dataset(dataset_id)
        
        if success:
            logger.info(f"User {user_id} deleted dataset {dataset_id}")
            return success_response(
                data={"deleted": True, "dataset_id": dataset_id},
                message="数据集删除成功"
            )
        else:
            return error_response(
                message="删除数据集失败",
                code=500,
                error_type="DELETE_DATASET_ERROR"
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
            error_type="DELETE_DATASET_ERROR"
        ), 500


# ============================================================================
# 数据集下载API
# ============================================================================

@dataset_detailed_bp.route('/<dataset_id>/download', methods=['GET'])
@jwt_required()
def download_dataset(dataset_id: str):
    """下载数据集文件
    
    生成数据集文件的下载链接或直接返回文件内容。
    支持生成临时下载URL或直接流式传输文件。
    需要JWT认证，且只能下载当前用户拥有的数据集。
    
    路径参数:
        dataset_id (str): 数据集唯一标识符 (UUID格式)
            - 必填
    
    查询参数:
        format (str): 下载格式 (可选)
            - 可选值: original, json, csv, parquet
            - 默认: original (原始格式)
        direct (bool): 是否直接下载 (可选)
            - true: 直接返回文件流
            - false: 返回下载URL
            - 默认: false
    
    请求头:
        Authorization (str): Bearer <access_token>
            - 必填
    
    响应体 (JSON) - 当direct=false时:
        成功 (200):
            {
                "code": 200,
                "message": "数据集下载准备完成",
                "data": {
                    "download_url": "string",        // 下载链接
                    "expires_at": "string",          // 链接过期时间
                    "file_name": "string",           // 文件名
                    "file_size": "integer",          // 文件大小(字节)
                    "format": "string"               // 文件格式
                }
            }
        失败:
            - 401: 未授权
            - 403: 无权下载此数据集
            - 404: 数据集不存在或文件不存在
            - 500: 服务器内部错误
    
    响应 - 当direct=true时:
        成功: 直接返回文件流
        失败: JSON错误响应
    
    示例:
        curl -X GET "http://localhost:5000/api/v1/dataset-detailed/xxx/download?format=json" \\
             -H "Authorization: Bearer <token>"
    """
    try:
        user_id = get_jwt_identity()
        download_format = request.args.get('format', 'original')
        direct = request.args.get('direct', 'false').lower() == 'true'
        
        # 获取数据集验证权限
        dataset = dataset_service.get_dataset(dataset_id)
        if not dataset:
            return error_response(
                message=f"数据集 {dataset_id} 不存在",
                code=404,
                error_type="DATASET_NOT_FOUND"
            ), 404
        
        if dataset.user_id != user_id:
            return error_response(
                message="无权下载此数据集",
                code=403,
                error_type="FORBIDDEN"
            ), 403
        
        # 生成下载信息
        download_info = dataset_service.generate_download_url(
            dataset_id=dataset_id,
            user_id=user_id,
            format=download_format
        )
        
        # 记录访问日志
        dataset_service.log_access(dataset_id, user_id, "download")
        
        # 如果是直接下载且文件存在
        if direct and download_info.get('file_path'):
            file_path = download_info['file_path']
            if os.path.exists(file_path):
                return send_file(
                    file_path,
                    as_attachment=True,
                    download_name=download_info.get('file_name', f'dataset_{dataset_id}.{download_format}')
                )
        
        return success_response(
            data=download_info,
            message="数据集下载准备完成"
        )
        
    except DatasetNotFoundError as e:
        return error_response(
            message=str(e),
            code=404,
            error_type="DATASET_NOT_FOUND"
        ), 404
    except Exception as e:
        logger.error(f"Error preparing download for dataset {dataset_id}: {e}")
        return error_response(
            message=f"下载数据集失败: {str(e)}",
            code=500,
            error_type="DOWNLOAD_ERROR"
        ), 500


# ============================================================================
# 数据集预览API
# ============================================================================

@dataset_detailed_bp.route('/<dataset_id>/preview', methods=['GET'])
@jwt_required()
def preview_dataset(dataset_id: str):
    """预览数据集内容
    
    获取数据集的前N行数据进行预览，支持分页和列过滤。
    不会加载整个数据集到内存，使用流式读取方式。
    需要JWT认证，且只能预览当前用户拥有的数据集。
    
    路径参数:
        dataset_id (str): 数据集唯一标识符 (UUID格式)
            - 必填
    
    查询参数:
        limit (int): 预览行数 (可选)
            - 范围: 1-100
            - 默认: 10
        offset (int): 起始偏移量 (可选)
            - 范围: >= 0
            - 默认: 0
        columns (str): 要显示的列名，逗号分隔 (可选)
            - 示例: "id,name,value"
            - 默认: 显示所有列
    
    请求头:
        Authorization (str): Bearer <access_token>
            - 必填
    
    响应体 (JSON):
        成功 (200):
            {
                "code": 200,
                "message": "数据集预览成功",
                "data": {
                    "dataset_id": "string",          // 数据集ID
                    "preview_data": [                // 预览数据数组
                        {
                            "column1": "value1",
                            "column2": "value2"
                        }
                    ],
                    "columns": ["string"],           // 列名列表
                    "column_types": {                // 列类型映射
                        "column1": "string",
                        "column2": "integer"
                    },
                    "total_rows": "integer",         // 数据集总行数
                    "preview_rows": "integer",       // 预览返回的行数
                    "limit": "integer",              // 请求的limit
                    "offset": "integer",             // 请求的offset
                    "has_more": "boolean"            // 是否还有更多数据
                }
            }
        失败:
            - 400: 查询参数无效
            - 401: 未授权
            - 403: 无权访问此数据集
            - 404: 数据集不存在
            - 500: 服务器内部错误
    
    示例:
        curl -X GET "http://localhost:5000/api/v1/dataset-detailed/xxx/preview?limit=20&offset=0" \\
             -H "Authorization: Bearer <token>"
    """
    try:
        user_id = get_jwt_identity()
        
        # 获取查询参数
        limit = min(int(request.args.get('limit', 10)), 100)
        offset = max(int(request.args.get('offset', 0)), 0)
        columns_param = request.args.get('columns')
        columns = columns_param.split(',') if columns_param else None
        
        # 获取数据集验证权限
        dataset = dataset_service.get_dataset(dataset_id)
        if not dataset:
            return error_response(
                message=f"数据集 {dataset_id} 不存在",
                code=404,
                error_type="DATASET_NOT_FOUND"
            ), 404
        
        if dataset.user_id != user_id:
            return error_response(
                message="无权访问此数据集",
                code=403,
                error_type="FORBIDDEN"
            ), 403
        
        # 获取预览数据
        preview_result = dataset_service.preview_dataset(
            dataset_id=dataset_id,
            limit=limit,
            offset=offset,
            columns=columns
        )
        
        # 记录访问日志
        dataset_service.log_access(dataset_id, user_id, "preview")
        
        return success_response(
            data=preview_result,
            message="数据集预览成功"
        )
        
    except ValueError as e:
        return error_response(
            message=f"查询参数无效: {str(e)}",
            code=400,
            error_type="VALIDATION_ERROR"
        ), 400
    except DatasetNotFoundError as e:
        return error_response(
            message=str(e),
            code=404,
            error_type="DATASET_NOT_FOUND"
        ), 404
    except Exception as e:
        logger.error(f"Error previewing dataset {dataset_id}: {e}")
        return error_response(
            message=f"预览数据集失败: {str(e)}",
            code=500,
            error_type="PREVIEW_ERROR"
        ), 500


# ============================================================================
# 数据集分析API
# ============================================================================

@dataset_detailed_bp.route('/<dataset_id>/analyze', methods=['POST'])
@jwt_required()
def analyze_dataset(dataset_id: str):
    """分析数据集
    
    对数据集进行统计分析，包括基本统计、数据质量分析和改进建议。
    支持多种分析级别，从基础统计到完整深度分析。
    需要JWT认证，且只能分析当前用户拥有的数据集。
    
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
            "analysis_type": "string",   // 分析类型 (可选)
                                         // 可选值: basic, detailed, full
                                         // 默认: basic
            "columns": ["string"],       // 要分析的列 (可选)
                                         // 默认: 分析所有列
            "sample_size": "integer",    // 采样大小 (可选)
                                         // 用于大数据集的快速分析
                                         // 默认: null (分析全部数据)
            "include_distributions": "boolean"  // 是否包含分布分析 (可选)
                                                // 默认: false
        }
    
    响应体 (JSON):
        成功 (200):
            {
                "code": 200,
                "message": "数据集分析完成",
                "data": {
                    "dataset_id": "string",
                    "analysis_type": "string",
                    "analyzed_at": "string",         // 分析时间 (ISO8601)
                    "basic_stats": {                 // 基本统计
                        "total_rows": "integer",
                        "total_columns": "integer",
                        "missing_values": "integer",
                        "duplicate_rows": "integer",
                        "memory_usage": "string"
                    },
                    "detailed_stats": {              // 详细统计 (analysis_type >= detailed)
                        "column_stats": [
                            {
                                "column_name": "string",
                                "data_type": "string",
                                "unique_values": "integer",
                                "missing_count": "integer",
                                "missing_percentage": "float",
                                "min": "any",
                                "max": "any",
                                "mean": "float",
                                "median": "float",
                                "std": "float"
                            }
                        ]
                    },
                    "data_quality": {                // 数据质量 (analysis_type >= detailed)
                        "completeness": "float",
                        "uniqueness": "float",
                        "consistency": "float",
                        "overall_score": "float"
                    },
                    "recommendations": [             // 改进建议 (analysis_type == full)
                        {
                            "type": "string",
                            "message": "string",
                            "priority": "string",    // high, medium, low
                            "affected_columns": ["string"]
                        }
                    ]
                }
            }
        失败:
            - 400: 请求数据无效
            - 401: 未授权
            - 403: 无权访问此数据集
            - 404: 数据集不存在
            - 500: 服务器内部错误
    
    示例:
        curl -X POST "http://localhost:5000/api/v1/dataset-detailed/xxx/analyze" \\
             -H "Authorization: Bearer <token>" \\
             -H "Content-Type: application/json" \\
             -d '{"analysis_type": "full"}'
    """
    try:
        user_id = get_jwt_identity()
        data = request.get_json() or {}
        
        analysis_type = data.get('analysis_type', 'basic')
        columns = data.get('columns')
        sample_size = data.get('sample_size')
        include_distributions = data.get('include_distributions', False)
        
        # 验证分析类型
        if analysis_type not in ['basic', 'detailed', 'full']:
            return error_response(
                message="无效的分析类型，支持: basic, detailed, full",
                code=400,
                error_type="VALIDATION_ERROR"
            ), 400
        
        # 获取数据集验证权限
        dataset = dataset_service.get_dataset(dataset_id)
        if not dataset:
            return error_response(
                message=f"数据集 {dataset_id} 不存在",
                code=404,
                error_type="DATASET_NOT_FOUND"
            ), 404
        
        if dataset.user_id != user_id:
            return error_response(
                message="无权访问此数据集",
                code=403,
                error_type="FORBIDDEN"
            ), 403
        
        # 执行分析
        analysis_result = dataset_service.analyze_dataset(
            dataset_id=dataset_id,
            analysis_type=analysis_type,
            columns=columns,
            sample_size=sample_size,
            include_distributions=include_distributions
        )
        
        # 记录访问日志
        dataset_service.log_access(dataset_id, user_id, "analyze")
        
        return success_response(
            data=analysis_result,
            message="数据集分析完成"
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
        logger.error(f"Error analyzing dataset {dataset_id}: {e}")
        return error_response(
            message=f"分析数据集失败: {str(e)}",
            code=500,
            error_type="ANALYSIS_ERROR"
        ), 500


# ============================================================================
# 数据集分割API
# ============================================================================

@dataset_detailed_bp.route('/<dataset_id>/split', methods=['POST'])
@jwt_required()
def split_dataset(dataset_id: str):
    """分割数据集
    
    将数据集按指定比例分割为训练集、验证集和测试集。
    支持分层采样和随机打乱，可选择创建新数据集或返回分割索引。
    需要JWT认证，且只能分割当前用户拥有的数据集。
    
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
            "split_ratios": {            // 分割比例 (可选)
                "train": "float",        // 训练集比例，默认0.8
                "validation": "float",   // 验证集比例，默认0.1
                "test": "float"          // 测试集比例，默认0.1
            },
            "shuffle": "boolean",        // 是否打乱数据 (可选，默认true)
            "seed": "integer",           // 随机种子 (可选，默认42)
            "stratify_column": "string", // 分层采样的目标列 (可选)
                                         // 用于保持类别比例
            "create_new_datasets": "boolean"  // 是否创建新数据集 (可选，默认true)
        }
    
    响应体 (JSON):
        成功 (200):
            {
                "code": 200,
                "message": "数据集分割完成",
                "data": {
                    "dataset_id": "string",
                    "split_config": {                // 分割配置
                        "ratios": {},
                        "shuffle": "boolean",
                        "seed": "integer",
                        "stratified": "boolean"
                    },
                    "splits": {                      // 分割结果
                        "train": {
                            "dataset_id": "string",  // 新数据集ID (如果创建)
                            "name": "string",
                            "size": "integer",
                            "samples": "integer"
                        },
                        "validation": {
                            "dataset_id": "string",
                            "name": "string",
                            "size": "integer",
                            "samples": "integer"
                        },
                        "test": {
                            "dataset_id": "string",
                            "name": "string",
                            "size": "integer",
                            "samples": "integer"
                        }
                    },
                    "total_samples": "integer",
                    "split_at": "string"             // 分割时间 (ISO8601)
                }
            }
        失败:
            - 400: 请求数据无效或分割比例错误
            - 401: 未授权
            - 403: 无权访问此数据集
            - 404: 数据集不存在
            - 500: 服务器内部错误
    
    示例:
        curl -X POST "http://localhost:5000/api/v1/dataset-detailed/xxx/split" \\
             -H "Authorization: Bearer <token>" \\
             -H "Content-Type: application/json" \\
             -d '{"split_ratios": {"train": 0.7, "validation": 0.15, "test": 0.15}}'
    """
    try:
        user_id = get_jwt_identity()
        data = request.get_json() or {}
        
        split_ratios = data.get('split_ratios', {
            'train': 0.8,
            'validation': 0.1,
            'test': 0.1
        })
        shuffle = data.get('shuffle', True)
        seed = data.get('seed', 42)
        stratify_column = data.get('stratify_column')
        create_new_datasets = data.get('create_new_datasets', True)
        
        # 验证分割比例
        total_ratio = sum(split_ratios.values())
        if abs(total_ratio - 1.0) > 0.01:
            return error_response(
                message=f"分割比例之和必须为1.0，当前为{total_ratio}",
                code=400,
                error_type="VALIDATION_ERROR"
            ), 400
        
        # 获取数据集验证权限
        dataset = dataset_service.get_dataset(dataset_id)
        if not dataset:
            return error_response(
                message=f"数据集 {dataset_id} 不存在",
                code=404,
                error_type="DATASET_NOT_FOUND"
            ), 404
        
        if dataset.user_id != user_id:
            return error_response(
                message="无权访问此数据集",
                code=403,
                error_type="FORBIDDEN"
            ), 403
        
        # 执行分割
        split_result = dataset_service.split_dataset(
            dataset_id=dataset_id,
            user_id=user_id,
            split_ratios=split_ratios,
            shuffle=shuffle,
            seed=seed,
            stratify_column=stratify_column,
            create_new_datasets=create_new_datasets
        )
        
        # 记录访问日志
        dataset_service.log_access(dataset_id, user_id, "split")
        
        return success_response(
            data=split_result,
            message="数据集分割完成"
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
        logger.error(f"Error splitting dataset {dataset_id}: {e}")
        return error_response(
            message=f"分割数据集失败: {str(e)}",
            code=500,
            error_type="SPLIT_ERROR"
        ), 500


# ============================================================================
# 数据集统计API
# ============================================================================

@dataset_detailed_bp.route('/<dataset_id>/statistics', methods=['GET'])
@jwt_required()
def get_dataset_statistics(dataset_id: str):
    """获取数据集统计信息
    
    获取数据集的详细统计信息，包括记录数、大小、列统计等。
    需要JWT认证，且只能获取当前用户拥有的数据集的统计信息。
    
    路径参数:
        dataset_id (str): 数据集唯一标识符 (UUID格式)
            - 必填
    
    请求头:
        Authorization (str): Bearer <access_token>
            - 必填
    
    响应体 (JSON):
        成功 (200):
            {
                "code": 200,
                "message": "获取统计信息成功",
                "data": {
                    "dataset_id": "string",
                    "total_rows": "integer",
                    "total_columns": "integer",
                    "size_bytes": "integer",
                    "size_human": "string",          // 人类可读的大小 (如 "1.5 MB")
                    "column_count_by_type": {
                        "integer": "integer",
                        "float": "integer",
                        "string": "integer",
                        "datetime": "integer"
                    },
                    "missing_values_total": "integer",
                    "missing_percentage": "float",
                    "duplicate_rows": "integer",
                    "last_analyzed": "string"        // 最后分析时间 (ISO8601)
                }
            }
        失败:
            - 401: 未授权
            - 403: 无权访问此数据集
            - 404: 数据集不存在
            - 500: 服务器内部错误
    
    示例:
        curl -X GET "http://localhost:5000/api/v1/dataset-detailed/xxx/statistics" \\
             -H "Authorization: Bearer <token>"
    """
    try:
        user_id = get_jwt_identity()
        
        # 获取数据集验证权限
        dataset = dataset_service.get_dataset(dataset_id)
        if not dataset:
            return error_response(
                message=f"数据集 {dataset_id} 不存在",
                code=404,
                error_type="DATASET_NOT_FOUND"
            ), 404
        
        if dataset.user_id != user_id:
            return error_response(
                message="无权访问此数据集",
                code=403,
                error_type="FORBIDDEN"
            ), 403
        
        # 获取统计信息
        statistics = dataset_service.get_detailed_statistics(dataset_id)
        
        return success_response(
            data=statistics,
            message="获取统计信息成功"
        )
        
    except DatasetNotFoundError as e:
        return error_response(
            message=str(e),
            code=404,
            error_type="DATASET_NOT_FOUND"
        ), 404
    except Exception as e:
        logger.error(f"Error getting statistics for dataset {dataset_id}: {e}")
        return error_response(
            message=f"获取统计信息失败: {str(e)}",
            code=500,
            error_type="GET_STATISTICS_ERROR"
        ), 500


# ============================================================================
# 数据集标签管理API
# ============================================================================

@dataset_detailed_bp.route('/<dataset_id>/tags', methods=['GET'])
@jwt_required()
def get_dataset_tags(dataset_id: str):
    """获取数据集标签列表
    
    获取指定数据集的所有标签。
    需要JWT认证，且只能获取当前用户拥有的数据集的标签。
    
    路径参数:
        dataset_id (str): 数据集唯一标识符 (UUID格式)
            - 必填
    
    请求头:
        Authorization (str): Bearer <access_token>
            - 必填
    
    响应体 (JSON):
        成功 (200):
            {
                "code": 200,
                "message": "获取标签成功",
                "data": {
                    "dataset_id": "string",
                    "tags": ["string"],
                    "count": "integer"
                }
            }
        失败:
            - 401: 未授权
            - 403: 无权访问此数据集
            - 404: 数据集不存在
            - 500: 服务器内部错误
    """
    try:
        user_id = get_jwt_identity()
        
        # 获取数据集验证权限
        dataset = dataset_service.get_dataset(dataset_id)
        if not dataset:
            return error_response(
                message=f"数据集 {dataset_id} 不存在",
                code=404,
                error_type="DATASET_NOT_FOUND"
            ), 404
        
        if dataset.user_id != user_id:
            return error_response(
                message="无权访问此数据集",
                code=403,
                error_type="FORBIDDEN"
            ), 403
        
        # 获取标签
        tags = dataset_service.get_dataset_tags(dataset_id)
        
        return success_response(
            data={
                "dataset_id": dataset_id,
                "tags": tags,
                "count": len(tags)
            },
            message="获取标签成功"
        )
        
    except Exception as e:
        logger.error(f"Error getting tags for dataset {dataset_id}: {e}")
        return error_response(
            message=f"获取标签失败: {str(e)}",
            code=500,
            error_type="GET_TAGS_ERROR"
        ), 500


@dataset_detailed_bp.route('/<dataset_id>/tags', methods=['POST'])
@jwt_required()
def add_dataset_tag(dataset_id: str):
    """添加数据集标签
    
    为指定数据集添加一个或多个标签。
    需要JWT认证，且只能为当前用户拥有的数据集添加标签。
    
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
            "tags": ["string"]           // 要添加的标签列表
                                         // 每个标签长度: 1-50字符
        }
    
    响应体 (JSON):
        成功 (200):
            {
                "code": 200,
                "message": "标签添加成功",
                "data": {
                    "dataset_id": "string",
                    "added_tags": ["string"],
                    "all_tags": ["string"]
                }
            }
        失败:
            - 400: 请求数据无效
            - 401: 未授权
            - 403: 无权访问此数据集
            - 404: 数据集不存在
            - 500: 服务器内部错误
    """
    try:
        user_id = get_jwt_identity()
        data = request.get_json() or {}
        
        tags = data.get('tags', [])
        if not tags or not isinstance(tags, list):
            return error_response(
                message="请提供有效的标签列表",
                code=400,
                error_type="VALIDATION_ERROR"
            ), 400
        
        # 获取数据集验证权限
        dataset = dataset_service.get_dataset(dataset_id)
        if not dataset:
            return error_response(
                message=f"数据集 {dataset_id} 不存在",
                code=404,
                error_type="DATASET_NOT_FOUND"
            ), 404
        
        if dataset.user_id != user_id:
            return error_response(
                message="无权修改此数据集",
                code=403,
                error_type="FORBIDDEN"
            ), 403
        
        # 添加标签
        added_tags = []
        for tag in tags:
            if tag and len(tag) <= 50:
                dataset_service.add_dataset_tag(dataset_id, tag, user_id)
                added_tags.append(tag)
        
        # 获取所有标签
        all_tags = dataset_service.get_dataset_tags(dataset_id)
        
        return success_response(
            data={
                "dataset_id": dataset_id,
                "added_tags": added_tags,
                "all_tags": all_tags
            },
            message="标签添加成功"
        )
        
    except Exception as e:
        logger.error(f"Error adding tags to dataset {dataset_id}: {e}")
        return error_response(
            message=f"添加标签失败: {str(e)}",
            code=500,
            error_type="ADD_TAGS_ERROR"
        ), 500


@dataset_detailed_bp.route('/<dataset_id>/tags/<tag>', methods=['DELETE'])
@jwt_required()
def remove_dataset_tag(dataset_id: str, tag: str):
    """删除数据集标签
    
    删除指定数据集的一个标签。
    需要JWT认证，且只能删除当前用户拥有的数据集的标签。
    
    路径参数:
        dataset_id (str): 数据集唯一标识符 (UUID格式)
            - 必填
        tag (str): 要删除的标签名
            - 必填
    
    请求头:
        Authorization (str): Bearer <access_token>
            - 必填
    
    响应体 (JSON):
        成功 (200):
            {
                "code": 200,
                "message": "标签删除成功",
                "data": {
                    "dataset_id": "string",
                    "removed_tag": "string",
                    "remaining_tags": ["string"]
                }
            }
        失败:
            - 401: 未授权
            - 403: 无权访问此数据集
            - 404: 数据集或标签不存在
            - 500: 服务器内部错误
    """
    try:
        user_id = get_jwt_identity()
        
        # 获取数据集验证权限
        dataset = dataset_service.get_dataset(dataset_id)
        if not dataset:
            return error_response(
                message=f"数据集 {dataset_id} 不存在",
                code=404,
                error_type="DATASET_NOT_FOUND"
            ), 404
        
        if dataset.user_id != user_id:
            return error_response(
                message="无权修改此数据集",
                code=403,
                error_type="FORBIDDEN"
            ), 403
        
        # 删除标签
        success = dataset_service.remove_dataset_tag(dataset_id, tag)
        
        # 获取剩余标签
        remaining_tags = dataset_service.get_dataset_tags(dataset_id)
        
        return success_response(
            data={
                "dataset_id": dataset_id,
                "removed_tag": tag,
                "remaining_tags": remaining_tags
            },
            message="标签删除成功"
        )
        
    except Exception as e:
        logger.error(f"Error removing tag from dataset {dataset_id}: {e}")
        return error_response(
            message=f"删除标签失败: {str(e)}",
            code=500,
            error_type="REMOVE_TAG_ERROR"
        ), 500


# ============================================================================
# 数据集版本管理API
# ============================================================================

@dataset_detailed_bp.route('/<dataset_id>/versions', methods=['GET'])
@jwt_required()
def get_dataset_versions(dataset_id: str):
    """获取数据集版本历史
    
    获取指定数据集的所有版本记录。
    需要JWT认证，且只能获取当前用户拥有的数据集的版本历史。
    
    路径参数:
        dataset_id (str): 数据集唯一标识符 (UUID格式)
            - 必填
    
    查询参数:
        limit (int): 返回数量限制 (可选，默认20)
        offset (int): 偏移量 (可选，默认0)
    
    请求头:
        Authorization (str): Bearer <access_token>
            - 必填
    
    响应体 (JSON):
        成功 (200):
            {
                "code": 200,
                "message": "获取版本历史成功",
                "data": {
                    "dataset_id": "string",
                    "versions": [
                        {
                            "version_id": "string",
                            "version": "string",
                            "description": "string",
                            "created_at": "string",
                            "created_by": "string",
                            "size": "integer",
                            "record_count": "integer",
                            "is_current": "boolean"
                        }
                    ],
                    "total": "integer",
                    "limit": "integer",
                    "offset": "integer"
                }
            }
        失败:
            - 401: 未授权
            - 403: 无权访问此数据集
            - 404: 数据集不存在
            - 500: 服务器内部错误
    """
    try:
        user_id = get_jwt_identity()
        limit = min(int(request.args.get('limit', 20)), 100)
        offset = max(int(request.args.get('offset', 0)), 0)
        
        # 获取数据集验证权限
        dataset = dataset_service.get_dataset(dataset_id)
        if not dataset:
            return error_response(
                message=f"数据集 {dataset_id} 不存在",
                code=404,
                error_type="DATASET_NOT_FOUND"
            ), 404
        
        if dataset.user_id != user_id:
            return error_response(
                message="无权访问此数据集",
                code=403,
                error_type="FORBIDDEN"
            ), 403
        
        # 获取版本历史
        versions_result = dataset_service.get_dataset_versions(
            dataset_id=dataset_id,
            limit=limit,
            offset=offset
        )
        
        return success_response(
            data=versions_result,
            message="获取版本历史成功"
        )
        
    except Exception as e:
        logger.error(f"Error getting versions for dataset {dataset_id}: {e}")
        return error_response(
            message=f"获取版本历史失败: {str(e)}",
            code=500,
            error_type="GET_VERSIONS_ERROR"
        ), 500


@dataset_detailed_bp.route('/<dataset_id>/versions', methods=['POST'])
@jwt_required()
def create_dataset_version(dataset_id: str):
    """创建数据集新版本
    
    为指定数据集创建一个新版本。
    新版本会复制当前数据并记录版本信息。
    需要JWT认证，且只能为当前用户拥有的数据集创建版本。
    
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
            "version": "string",         // 版本号 (可选，自动递增)
            "description": "string",     // 版本描述 (可选)
            "changelog": "string"        // 变更日志 (可选)
        }
    
    响应体 (JSON):
        成功 (201):
            {
                "code": 201,
                "message": "版本创建成功",
                "data": {
                    "version_id": "string",
                    "dataset_id": "string",
                    "version": "string",
                    "description": "string",
                    "created_at": "string",
                    "created_by": "string"
                }
            }
        失败:
            - 400: 请求数据无效
            - 401: 未授权
            - 403: 无权访问此数据集
            - 404: 数据集不存在
            - 500: 服务器内部错误
    """
    try:
        user_id = get_jwt_identity()
        data = request.get_json() or {}
        
        version = data.get('version')
        description = data.get('description')
        changelog = data.get('changelog')
        
        # 获取数据集验证权限
        dataset = dataset_service.get_dataset(dataset_id)
        if not dataset:
            return error_response(
                message=f"数据集 {dataset_id} 不存在",
                code=404,
                error_type="DATASET_NOT_FOUND"
            ), 404
        
        if dataset.user_id != user_id:
            return error_response(
                message="无权修改此数据集",
                code=403,
                error_type="FORBIDDEN"
            ), 403
        
        # 创建新版本
        new_version = dataset_service.create_dataset_version(
            dataset_id=dataset_id,
            user_id=user_id,
            version=version,
            description=description,
            changelog=changelog
        )
        
        return success_response(
            data=new_version,
            message="版本创建成功"
        ), 201
        
    except DatasetValidationError as e:
        return error_response(
            message=str(e),
            code=400,
            error_type="VALIDATION_ERROR"
        ), 400
    except Exception as e:
        logger.error(f"Error creating version for dataset {dataset_id}: {e}")
        return error_response(
            message=f"创建版本失败: {str(e)}",
            code=500,
            error_type="CREATE_VERSION_ERROR"
        ), 500


# ============================================================================
# 健康检查API
# ============================================================================

@dataset_detailed_bp.route('/health', methods=['GET'])
def health_check():
    """健康检查
    
    检查数据集详细管理服务的运行状态。
    不需要认证。
    
    响应体 (JSON):
        成功 (200):
            {
                "code": 200,
                "message": "服务运行正常",
                "data": {
                    "status": "healthy",
                    "service": "dataset_detailed",
                    "version": "1.0.0",
                    "timestamp": "string"
                }
            }
    """
    return success_response(
        data={
            "status": "healthy",
            "service": "dataset_detailed",
            "version": "1.0.0",
            "timestamp": datetime.utcnow().isoformat()
        },
        message="服务运行正常"
    )

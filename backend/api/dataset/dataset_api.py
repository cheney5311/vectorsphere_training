"""数据集API接口

提供数据集相关的RESTful API接口。

模块说明:
    - 提供数据集的完整CRUD操作
    - 支持数据集搜索、过滤和分页
    - 支持标签管理和版本控制
    - 支持批量操作和统计查询
    - 所有接口需要JWT认证

路由前缀: /api/v1/datasets

作者: VectorSphere Team
版本: 1.0.0
"""

import sys
import os
import logging
from datetime import datetime
from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity

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
dataset_bp = Blueprint('dataset', __name__, url_prefix='/api/v1/datasets')

# 初始化服务
dataset_repository = DatasetRepository()
dataset_service = DatasetService(dataset_repository)


# ============================================================================
# 辅助函数
# ============================================================================

def check_dataset_permission(dataset, user_id: str) -> tuple:
    """检查数据集访问权限
    
    Args:
        dataset: 数据集对象
        user_id: 用户ID
        
    Returns:
        tuple: (是否有权限, 错误响应或None)
    """
    if not dataset:
        return False, error_response("数据集不存在", 404)
    if dataset.user_id != user_id:
        return False, error_response("无权访问此数据集", 403)
    return True, None


# ============================================================================
# 基础CRUD接口
# ============================================================================

@dataset_bp.route('', methods=['POST'])
@jwt_required()
def create_dataset():
    """创建数据集
    
    创建一个新的数据集。
    
    请求头:
        Authorization: Bearer <token> - JWT访问令牌
        Content-Type: application/json
        
    请求体:
        {
            "name": "string",           # 必填，数据集名称，1-200字符
            "description": "string",    # 可选，数据集描述
            "type": "string",           # 可选，数据集类型，默认"text"
                                        # 可选值: text, image, audio, video, tabular, mixed
            "format": "string",         # 可选，数据格式，默认"json"
                                        # 可选值: json, csv, parquet, tfrecord, arrow, custom
            "storage_path": "string",   # 可选，存储路径
            "config": {}                # 可选，配置信息，JSON对象
        }
        
    响应:
        201 Created:
            {
                "success": true,
                "message": "操作成功",
                "data": {
                    "dataset_id": "uuid",
                    "name": "string",
                    "status": "pending",
                    "created_at": "ISO8601时间戳",
                    ...
                }
            }
        400 Bad Request: 参数验证失败
        401 Unauthorized: 未授权
        500 Internal Server Error: 服务器错误
        
    示例:
        curl -X POST /api/v1/datasets \\
            -H "Authorization: Bearer <token>" \\
            -H "Content-Type: application/json" \\
            -d '{"name": "training_data", "type": "text"}'
    """
    try:
        # 获取当前用户ID
        user_id = get_jwt_identity()
        
        # 验证请求数据
        data = request.get_json()
        if not data:
            return error_response("请求数据不能为空", 400)
            
        # 提取参数
        name = data.get('name')
        if not name:
            return error_response("数据集名称不能为空", 400)
            
        description = data.get('description')
        dataset_type = data.get('type', 'text')
        format_type = data.get('format', 'json')
        storage_path = data.get('storage_path', '')
        config = data.get('config', {})
        
        # 创建数据集
        dataset = dataset_service.create_dataset(
            user_id=user_id,
            name=name,
            description=description,
            dataset_type=dataset_type,
            format=format_type,
            storage_path=storage_path,
            config=config
        )
        
        logger.info(f"User {user_id} created dataset {dataset.dataset_id}")
        return success_response(dataset.to_dict(), "创建成功", 201)
        
    except DatasetValidationError as e:
        logger.warning(f"Validation error: {e}")
        return error_response(str(e), 400)
    except ValidationError as e:
        logger.warning(f"Validation error: {e}")
        return error_response(str(e), 400)
    except Exception as e:
        logger.error(f"Failed to create dataset: {e}", exc_info=True)
        return error_response(f"创建数据集失败: {str(e)}", 500)


@dataset_bp.route('/<dataset_id>', methods=['GET'])
@jwt_required()
def get_dataset(dataset_id):
    """获取数据集详情
    
    根据ID获取数据集的详细信息。
    
    路径参数:
        dataset_id: string - 数据集唯一标识符(UUID)
        
    请求头:
        Authorization: Bearer <token>
        
    响应:
        200 OK:
            {
                "success": true,
                "data": {
                    "dataset_id": "uuid",
                    "user_id": "string",
                    "name": "string",
                    "description": "string",
                    "dataset_type": "string",
                    "format": "string",
                    "status": "string",
                    "storage_path": "string",
                    "size": number,
                    "record_count": number,
                    "features": {},
                    "labels": {},
                    "config": {},
                    "version": "string",
                    "checksum": "string",
                    "ready": boolean,
                    "validated": boolean,
                    "validation_result": {},
                    "created_at": "ISO8601时间戳",
                    "updated_at": "ISO8601时间戳"
                }
            }
        400 Bad Request: ID格式错误
        403 Forbidden: 无权限访问
        404 Not Found: 数据集不存在
        
    示例:
        curl -X GET /api/v1/datasets/550e8400-e29b-41d4-a716-446655440000 \\
            -H "Authorization: Bearer <token>"
    """
    try:
        user_id = get_jwt_identity()
        
        dataset = dataset_service.get_dataset(dataset_id, user_id)
        
        has_permission, error = check_dataset_permission(dataset, user_id)
        if not has_permission:
            return error
            
        return success_response(dataset.to_dict())
        
    except DatasetValidationError as e:
        return error_response(str(e), 400)
    except Exception as e:
        logger.error(f"Failed to get dataset {dataset_id}: {e}")
        return error_response(f"获取数据集失败: {str(e)}", 500)


@dataset_bp.route('', methods=['GET'])
@jwt_required()
def list_datasets():
    """获取数据集列表
    
    分页获取当前用户的数据集列表，支持过滤和排序。
    
    查询参数:
        limit: int - 返回数量限制，默认50，最大100
        offset: int - 偏移量，默认0
        status: string - 按状态过滤 (pending/uploading/processing/ready/error/archived)
        type: string - 按类型过滤 (text/image/audio/video/tabular/mixed)
        order_by: string - 排序字段 (created_at/updated_at/name/size)，默认created_at
        order_desc: boolean - 是否降序，默认true
        
    请求头:
        Authorization: Bearer <token>
        
    响应:
        200 OK:
            {
                "success": true,
                "data": {
                    "datasets": [
                        {
                            "dataset_id": "uuid",
                            "name": "string",
                            "status": "string",
                            ...
                        }
                    ],
                    "total": number,
                    "limit": number,
                    "offset": number,
                    "has_more": boolean
                }
            }
        400 Bad Request: 参数验证失败
        
    示例:
        curl -X GET "/api/v1/datasets?limit=10&status=ready&order_by=name" \\
            -H "Authorization: Bearer <token>"
    """
    try:
        user_id = get_jwt_identity()
        
        # 获取查询参数
        limit = request.args.get('limit', 50, type=int)
        offset = request.args.get('offset', 0, type=int)
        status = request.args.get('status')
        dataset_type = request.args.get('type')
        order_by = request.args.get('order_by', 'created_at')
        order_desc = request.args.get('order_desc', 'true').lower() == 'true'
        
        # 参数验证
        if limit <= 0 or limit > 100:
            return error_response("limit必须在1-100之间", 400)
        if offset < 0:
            return error_response("offset不能为负数", 400)
        
        # 获取数据集列表(带分页信息)
        response = dataset_service.list_datasets_with_pagination(
            user_id=user_id,
            limit=limit,
            offset=offset,
            status=status,
            dataset_type=dataset_type
        )
        
        return success_response(response.to_dict())
        
    except DatasetValidationError as e:
        return error_response(str(e), 400)
    except Exception as e:
        logger.error(f"Failed to list datasets: {e}")
        return error_response(f"获取数据集列表失败: {str(e)}", 500)


@dataset_bp.route('/<dataset_id>', methods=['PUT'])
@jwt_required()
def update_dataset(dataset_id):
    """更新数据集
    
    更新指定数据集的信息。
    
    路径参数:
        dataset_id: string - 数据集唯一标识符
        
    请求体:
        {
            "name": "string",           # 可选，新的名称
            "description": "string",    # 可选，新的描述
            "config": {},               # 可选，新的配置
            "status": "string"          # 可选，新的状态
        }
        
    响应:
        200 OK: 更新后的数据集信息
        400 Bad Request: 参数验证失败
        403 Forbidden: 无权限
        404 Not Found: 数据集不存在
        
    示例:
        curl -X PUT /api/v1/datasets/550e8400-e29b-41d4-a716-446655440000 \\
            -H "Authorization: Bearer <token>" \\
            -H "Content-Type: application/json" \\
            -d '{"name": "updated_name", "description": "更新描述"}'
    """
    try:
        user_id = get_jwt_identity()
        
        # 验证请求数据
        data = request.get_json()
        if not data:
            return error_response("请求数据不能为空", 400)
            
        # 权限检查
        dataset = dataset_service.get_dataset(dataset_id)
        has_permission, error = check_dataset_permission(dataset, user_id)
        if not has_permission:
            return error
            
        # 提取参数
        name = data.get('name')
        description = data.get('description')
        config = data.get('config')
        status = data.get('status')
        
        # 更新数据集
        updated_dataset = dataset_service.update_dataset(
            dataset_id=dataset_id,
            name=name,
            description=description,
            config=config,
            status=status,
            user_id=user_id
        )
        
        logger.info(f"User {user_id} updated dataset {dataset_id}")
        return success_response(updated_dataset.to_dict())
        
    except DatasetNotFoundError as e:
        return error_response(str(e), 404)
    except DatasetValidationError as e:
        return error_response(str(e), 400)
    except ValidationError as e:
        return error_response(str(e), 400)
    except Exception as e:
        logger.error(f"Failed to update dataset {dataset_id}: {e}")
        return error_response(f"更新数据集失败: {str(e)}", 500)


@dataset_bp.route('/<dataset_id>', methods=['DELETE'])
@jwt_required()
def delete_dataset(dataset_id):
    """删除数据集
    
    删除指定的数据集。
    
    路径参数:
        dataset_id: string - 数据集唯一标识符
        
    响应:
        200 OK: 删除成功
        400 Bad Request: ID格式错误或业务逻辑错误(如正在处理中)
        403 Forbidden: 无权限
        404 Not Found: 数据集不存在
        
    注意:
        - 处理中的数据集无法删除
        - 删除后数据不可恢复
        
    示例:
        curl -X DELETE /api/v1/datasets/550e8400-e29b-41d4-a716-446655440000 \\
            -H "Authorization: Bearer <token>"
    """
    try:
        user_id = get_jwt_identity()
        
        # 权限检查
        dataset = dataset_service.get_dataset(dataset_id)
        has_permission, error = check_dataset_permission(dataset, user_id)
        if not has_permission:
            return error
            
        # 删除数据集
        success = dataset_service.delete_dataset(dataset_id, user_id)
        
        if success:
            logger.info(f"User {user_id} deleted dataset {dataset_id}")
            return success_response({"deleted": True}, "删除成功")
        else:
            return error_response("删除数据集失败", 500)
            
    except DatasetBusinessLogicError as e:
        return error_response(str(e), 400)
    except DatasetValidationError as e:
        return error_response(str(e), 400)
    except Exception as e:
        logger.error(f"Failed to delete dataset {dataset_id}: {e}")
        return error_response(f"删除数据集失败: {str(e)}", 500)


# ============================================================================
# 状态管理接口
# ============================================================================

@dataset_bp.route('/<dataset_id>/process', methods=['POST'])
@jwt_required()
def process_dataset(dataset_id):
    """处理数据集
    
    将数据集状态设置为处理中，触发数据处理流程。
    
    路径参数:
        dataset_id: string - 数据集唯一标识符
        
    响应:
        200 OK: 更新后的数据集信息，status为"processing"
        400 Bad Request: 数据集已在处理中或已归档
        403 Forbidden: 无权限
        404 Not Found: 数据集不存在
        
    示例:
        curl -X POST /api/v1/datasets/550e8400-e29b-41d4-a716-446655440000/process \\
            -H "Authorization: Bearer <token>"
    """
    try:
        user_id = get_jwt_identity()
        
        # 权限检查
        dataset = dataset_service.get_dataset(dataset_id)
        has_permission, error = check_dataset_permission(dataset, user_id)
        if not has_permission:
            return error
            
        # 处理数据集
        processed_dataset = dataset_service.process_dataset(dataset_id, user_id)
        
        return success_response(processed_dataset.to_dict(), "开始处理")
        
    except DatasetNotFoundError as e:
        return error_response(str(e), 404)
    except DatasetBusinessLogicError as e:
        return error_response(str(e), 400)
    except DatasetValidationError as e:
        return error_response(str(e), 400)
    except Exception as e:
        logger.error(f"Failed to process dataset {dataset_id}: {e}")
        return error_response(f"处理数据集失败: {str(e)}", 500)


@dataset_bp.route('/<dataset_id>/validate', methods=['POST'])
@jwt_required()
def validate_dataset(dataset_id):
    """验证数据集
    
    记录数据集的验证结果。
    
    路径参数:
        dataset_id: string - 数据集唯一标识符
        
    请求体:
        {
            "validation_result": {       # 必填，验证结果
                "is_valid": boolean,     # 是否验证通过
                "errors": [],            # 错误列表
                "warnings": [],          # 警告列表
                "statistics": {}         # 统计信息
            }
        }
        
    响应:
        200 OK: 更新后的数据集信息
        400 Bad Request: 参数错误
        403 Forbidden: 无权限
        404 Not Found: 数据集不存在
        
    示例:
        curl -X POST /api/v1/datasets/550e8400-e29b-41d4-a716-446655440000/validate \\
            -H "Authorization: Bearer <token>" \\
            -H "Content-Type: application/json" \\
            -d '{"validation_result": {"is_valid": true, "errors": []}}'
    """
    try:
        user_id = get_jwt_identity()
        
        data = request.get_json()
        if not data:
            return error_response("请求数据不能为空", 400)
            
        validation_result = data.get('validation_result', {})
        
        # 权限检查
        dataset = dataset_service.get_dataset(dataset_id)
        has_permission, error = check_dataset_permission(dataset, user_id)
        if not has_permission:
            return error
            
        # 验证数据集
        validated_dataset = dataset_service.validate_dataset(
            dataset_id, 
            validation_result,
            user_id
        )
        
        return success_response(validated_dataset.to_dict(), "验证完成")
        
    except DatasetNotFoundError as e:
        return error_response(str(e), 404)
    except DatasetValidationError as e:
        return error_response(str(e), 400)
    except Exception as e:
        logger.error(f"Failed to validate dataset {dataset_id}: {e}")
        return error_response(f"验证数据集失败: {str(e)}", 500)


@dataset_bp.route('/<dataset_id>/ready', methods=['POST'])
@jwt_required()
def mark_dataset_ready(dataset_id):
    """标记数据集为就绪状态
    
    将数据集状态设置为ready，表示可以用于训练。
    
    路径参数:
        dataset_id: string - 数据集唯一标识符
        
    响应:
        200 OK: 更新后的数据集信息，status为"ready"，ready为true
        403 Forbidden: 无权限
        404 Not Found: 数据集不存在
        
    示例:
        curl -X POST /api/v1/datasets/550e8400-e29b-41d4-a716-446655440000/ready \\
            -H "Authorization: Bearer <token>"
    """
    try:
        user_id = get_jwt_identity()
        
        # 权限检查
        dataset = dataset_service.get_dataset(dataset_id)
        has_permission, error = check_dataset_permission(dataset, user_id)
        if not has_permission:
            return error
            
        # 标记就绪
        ready_dataset = dataset_service.mark_dataset_ready(dataset_id, user_id)
        
        return success_response(ready_dataset.to_dict(), "已标记为就绪")
        
    except DatasetNotFoundError as e:
        return error_response(str(e), 404)
    except DatasetValidationError as e:
        return error_response(str(e), 400)
    except Exception as e:
        logger.error(f"Failed to mark dataset ready {dataset_id}: {e}")
        return error_response(f"标记数据集就绪失败: {str(e)}", 500)


@dataset_bp.route('/<dataset_id>/archive', methods=['POST'])
@jwt_required()
def archive_dataset(dataset_id):
    """归档数据集
    
    将数据集状态设置为archived。
    
    路径参数:
        dataset_id: string - 数据集唯一标识符
        
    响应:
        200 OK: 更新后的数据集信息，status为"archived"
        
    示例:
        curl -X POST /api/v1/datasets/550e8400-e29b-41d4-a716-446655440000/archive \\
            -H "Authorization: Bearer <token>"
    """
    try:
        user_id = get_jwt_identity()
        
        # 权限检查
        dataset = dataset_service.get_dataset(dataset_id)
        has_permission, error = check_dataset_permission(dataset, user_id)
        if not has_permission:
            return error
            
        # 归档
        archived_dataset = dataset_service.archive_dataset(dataset_id, user_id)
        
        return success_response(archived_dataset.to_dict(), "已归档")
        
    except DatasetNotFoundError as e:
        return error_response(str(e), 404)
    except Exception as e:
        logger.error(f"Failed to archive dataset {dataset_id}: {e}")
        return error_response(f"归档数据集失败: {str(e)}", 500)


# ============================================================================
# 搜索和统计接口
# ============================================================================

@dataset_bp.route('/search', methods=['GET'])
@jwt_required()
def search_datasets():
    """搜索数据集
    
    支持关键字搜索和多条件过滤的数据集查询。
    
    查询参数:
        keyword: string - 搜索关键字，匹配名称和描述
        status: string - 状态过滤
        type: string - 类型过滤
        tags: string - 标签过滤，逗号分隔
        order_by: string - 排序字段 (created_at/updated_at/name/size)，默认created_at
        order_desc: bool - 是否降序，默认true
        limit: int - 返回数量限制，默认50
        offset: int - 偏移量，默认0
        
    响应:
        200 OK:
            {
                "success": true,
                "data": {
                    "datasets": [...],
                    "total": number,
                    "limit": number,
                    "offset": number,
                    "has_more": boolean
                }
            }
            
    示例:
        curl -X GET "/api/v1/datasets/search?keyword=training&status=ready&order_by=name" \\
            -H "Authorization: Bearer <token>"
    """
    try:
        user_id = get_jwt_identity()
        
        # 获取查询参数
        keyword = request.args.get('keyword')
        status = request.args.get('status')
        dataset_type = request.args.get('type')
        tags_str = request.args.get('tags')
        order_by = request.args.get('order_by', 'created_at')
        order_desc_str = request.args.get('order_desc', 'true')
        limit = request.args.get('limit', 50, type=int)
        offset = request.args.get('offset', 0, type=int)
        
        # 解析标签过滤
        tag_filter = [t.strip() for t in tags_str.split(',') if t.strip()] if tags_str else None
        
        # 解析排序方向
        order_desc = order_desc_str.lower() in ('true', '1', 'yes')
        
        # 搜索
        result = dataset_service.search_datasets(
            user_id=user_id,
            search=keyword,
            status=status,
            dataset_type=dataset_type,
            tag_filter=tag_filter,
            limit=limit,
            offset=offset,
            order_by=order_by,
            order_desc=order_desc
        )
        
        return success_response({
            'datasets': result['datasets'],
            'total': result['total_count'],
            'limit': result['limit'],
            'offset': result['offset'],
            'has_more': result['has_more']
        })
        
    except Exception as e:
        logger.error(f"Failed to search datasets: {e}")
        return error_response(f"搜索数据集失败: {str(e)}", 500)


@dataset_bp.route('/statistics', methods=['GET'])
@jwt_required()
def get_statistics():
    """获取数据集统计信息
    
    获取当前用户的数据集统计概览。
    
    响应:
        200 OK:
            {
                "success": true,
                "data": {
                    "total": number,          # 总数据集数
                    "by_status": {            # 按状态分组
                        "pending": number,
                        "ready": number,
                        ...
                    },
                    "by_type": {              # 按类型分组
                        "text": number,
                        "image": number,
                        ...
                    },
                    "total_size": number,     # 总存储大小(字节)
                    "total_records": number   # 总记录数
                }
            }
            
    示例:
        curl -X GET /api/v1/datasets/statistics \\
            -H "Authorization: Bearer <token>"
    """
    try:
        user_id = get_jwt_identity()
        
        stats = dataset_service.get_statistics(user_id)
        
        return success_response(stats)
        
    except Exception as e:
        logger.error(f"Failed to get statistics: {e}")
        return error_response(f"获取统计信息失败: {str(e)}", 500)


# ============================================================================
# 批量操作接口
# ============================================================================

@dataset_bp.route('/bulk/delete', methods=['POST'])
@jwt_required()
def bulk_delete_datasets():
    """批量删除数据集
    
    批量删除多个数据集。
    
    请求体:
        {
            "dataset_ids": ["uuid1", "uuid2", ...]  # 必填，数据集ID列表
        }
        
    响应:
        200 OK:
            {
                "success": true,
                "data": {
                    "deleted": number,    # 成功删除数量
                    "failed": number,     # 失败数量
                    "errors": []          # 错误信息
                }
            }
        400 Bad Request: 参数错误
        
    示例:
        curl -X POST /api/v1/datasets/bulk/delete \\
            -H "Authorization: Bearer <token>" \\
            -H "Content-Type: application/json" \\
            -d '{"dataset_ids": ["id1", "id2"]}'
    """
    try:
        user_id = get_jwt_identity()
        
        data = request.get_json()
        if not data:
            return error_response("请求数据不能为空", 400)
            
        dataset_ids = data.get('dataset_ids', [])
        if not dataset_ids:
            return error_response("dataset_ids不能为空", 400)
        if not isinstance(dataset_ids, list):
            return error_response("dataset_ids必须是数组", 400)
        
        # 验证所有数据集的权限
        for dataset_id in dataset_ids:
            dataset = dataset_service.get_dataset(dataset_id)
            if dataset and dataset.user_id != user_id:
                return error_response(f"无权删除数据集 {dataset_id}", 403)
        
        # 批量删除
        result = dataset_service.bulk_delete(dataset_ids, user_id)
        
        logger.info(f"User {user_id} bulk deleted {result['deleted']} datasets")
        return success_response(result)
        
    except Exception as e:
        logger.error(f"Failed to bulk delete datasets: {e}")
        return error_response(f"批量删除失败: {str(e)}", 500)


# ============================================================================
# 克隆接口
# ============================================================================

@dataset_bp.route('/<dataset_id>/clone', methods=['POST'])
@jwt_required()
def clone_dataset(dataset_id):
    """克隆数据集
    
    创建现有数据集的副本。
    
    路径参数:
        dataset_id: string - 源数据集ID
        
    请求体:
        {
            "name": "string"   # 可选，新数据集名称，默认原名称+"_copy"
        }
        
    响应:
        201 Created: 新创建的数据集信息
        403 Forbidden: 无权限
        404 Not Found: 源数据集不存在
        
    示例:
        curl -X POST /api/v1/datasets/550e8400-e29b-41d4-a716-446655440000/clone \\
            -H "Authorization: Bearer <token>" \\
            -H "Content-Type: application/json" \\
            -d '{"name": "my_copy"}'
    """
    try:
        user_id = get_jwt_identity()
        
        # 权限检查
        dataset = dataset_service.get_dataset(dataset_id)
        has_permission, error = check_dataset_permission(dataset, user_id)
        if not has_permission:
            return error
        
        # 获取新名称
        data = request.get_json() or {}
        new_name = data.get('name')
        
        # 克隆
        cloned = dataset_service.clone_dataset(
            source_dataset_id=dataset_id,
            user_id=user_id,
            new_name=new_name
        )
        
        logger.info(f"User {user_id} cloned dataset {dataset_id} to {cloned.dataset_id}")
        return success_response(cloned.to_dict(), "克隆成功", 201)
        
    except DatasetNotFoundError as e:
        return error_response(str(e), 404)
    except Exception as e:
        logger.error(f"Failed to clone dataset {dataset_id}: {e}")
        return error_response(f"克隆数据集失败: {str(e)}", 500)


# ============================================================================
# 标签管理接口
# ============================================================================

@dataset_bp.route('/<dataset_id>/tags', methods=['GET'])
@jwt_required()
def get_dataset_tags(dataset_id):
    """获取数据集标签
    
    获取数据集的所有标签。
    
    路径参数:
        dataset_id: string - 数据集ID
        
    响应:
        200 OK:
            {
                "success": true,
                "data": {
                    "tags": [
                        {
                            "tag_id": "uuid",
                            "tag_name": "string",
                            "tag_value": "string"
                        }
                    ]
                }
            }
            
    示例:
        curl -X GET /api/v1/datasets/550e8400-e29b-41d4-a716-446655440000/tags \\
            -H "Authorization: Bearer <token>"
    """
    try:
        user_id = get_jwt_identity()
        
        # 权限检查
        dataset = dataset_service.get_dataset(dataset_id)
        has_permission, error = check_dataset_permission(dataset, user_id)
        if not has_permission:
            return error
        
        tags = dataset_service.get_tags(dataset_id)
        
        return success_response({'tags': tags})
        
    except DatasetNotFoundError as e:
        return error_response(str(e), 404)
    except Exception as e:
        logger.error(f"Failed to get tags for dataset {dataset_id}: {e}")
        return error_response(f"获取标签失败: {str(e)}", 500)


@dataset_bp.route('/<dataset_id>/tags', methods=['POST'])
@jwt_required()
def add_dataset_tag(dataset_id):
    """添加数据集标签
    
    为数据集添加新标签。
    
    路径参数:
        dataset_id: string - 数据集ID
        
    请求体:
        {
            "tag_name": "string",   # 必填，标签名称
            "tag_value": "string"   # 可选，标签值
        }
        
    响应:
        201 Created:
            {
                "success": true,
                "data": {
                    "tag_id": "uuid"
                }
            }
            
    示例:
        curl -X POST /api/v1/datasets/550e8400-e29b-41d4-a716-446655440000/tags \\
            -H "Authorization: Bearer <token>" \\
            -H "Content-Type: application/json" \\
            -d '{"tag_name": "category", "tag_value": "training"}'
    """
    try:
        user_id = get_jwt_identity()
        
        # 权限检查
        dataset = dataset_service.get_dataset(dataset_id)
        has_permission, error = check_dataset_permission(dataset, user_id)
        if not has_permission:
            return error
        
        data = request.get_json()
        if not data:
            return error_response("请求数据不能为空", 400)
            
        tag_name = data.get('tag_name')
        if not tag_name:
            return error_response("tag_name不能为空", 400)
            
        tag_value = data.get('tag_value')
        
        tag_id = dataset_service.add_tag(
            dataset_id=dataset_id,
            tag_name=tag_name,
            tag_value=tag_value,
            user_id=user_id
        )
        
        return success_response({'tag_id': tag_id}, "添加成功", 201)
        
    except DatasetNotFoundError as e:
        return error_response(str(e), 404)
    except Exception as e:
        logger.error(f"Failed to add tag to dataset {dataset_id}: {e}")
        return error_response(f"添加标签失败: {str(e)}", 500)


@dataset_bp.route('/<dataset_id>/tags/<tag_name>', methods=['DELETE'])
@jwt_required()
def remove_dataset_tag(dataset_id, tag_name):
    """移除数据集标签
    
    移除数据集的指定标签。
    
    路径参数:
        dataset_id: string - 数据集ID
        tag_name: string - 标签名称
        
    响应:
        200 OK: 移除成功
        404 Not Found: 数据集或标签不存在
        
    示例:
        curl -X DELETE /api/v1/datasets/550e8400-e29b-41d4-a716-446655440000/tags/category \\
            -H "Authorization: Bearer <token>"
    """
    try:
        user_id = get_jwt_identity()
        
        # 权限检查
        dataset = dataset_service.get_dataset(dataset_id)
        has_permission, error = check_dataset_permission(dataset, user_id)
        if not has_permission:
            return error
        
        success = dataset_service.remove_tag(dataset_id, tag_name)
        
        if success:
            return success_response({'removed': True}, "移除成功")
        else:
            return error_response("标签不存在", 404)
            
    except DatasetNotFoundError as e:
        return error_response(str(e), 404)
    except Exception as e:
        logger.error(f"Failed to remove tag from dataset {dataset_id}: {e}")
        return error_response(f"移除标签失败: {str(e)}", 500)


@dataset_bp.route('/by-tag', methods=['GET'])
@jwt_required()
def find_datasets_by_tag():
    """按标签查找数据集
    
    根据标签查找数据集。
    
    查询参数:
        tag_name: string - 必填，标签名称
        tag_value: string - 可选，标签值
        
    响应:
        200 OK:
            {
                "success": true,
                "data": {
                    "datasets": [...]
                }
            }
            
    示例:
        curl -X GET "/api/v1/datasets/by-tag?tag_name=category&tag_value=training" \\
            -H "Authorization: Bearer <token>"
    """
    try:
        user_id = get_jwt_identity()
        
        tag_name = request.args.get('tag_name')
        if not tag_name:
            return error_response("tag_name参数必填", 400)
            
        tag_value = request.args.get('tag_value')
        
        datasets = dataset_service.find_by_tag(user_id, tag_name, tag_value)
        
        return success_response({
            'datasets': [d.to_dict() for d in datasets]
        })
        
    except Exception as e:
        logger.error(f"Failed to find datasets by tag: {e}")
        return error_response(f"查找失败: {str(e)}", 500)


# ============================================================================
# 版本管理接口
# ============================================================================

@dataset_bp.route('/<dataset_id>/versions', methods=['GET'])
@jwt_required()
def get_dataset_versions(dataset_id):
    """获取数据集版本列表
    
    获取数据集的所有版本历史。
    
    路径参数:
        dataset_id: string - 数据集ID
        
    响应:
        200 OK:
            {
                "success": true,
                "data": {
                    "versions": [
                        {
                            "version_id": "uuid",
                            "version": "string",
                            "description": "string",
                            "created_at": "ISO8601",
                            "created_by": "string",
                            "size": number,
                            "record_count": number
                        }
                    ]
                }
            }
            
    示例:
        curl -X GET /api/v1/datasets/550e8400-e29b-41d4-a716-446655440000/versions \\
            -H "Authorization: Bearer <token>"
    """
    try:
        user_id = get_jwt_identity()
        
        # 权限检查
        dataset = dataset_service.get_dataset(dataset_id)
        has_permission, error = check_dataset_permission(dataset, user_id)
        if not has_permission:
            return error
        
        versions = dataset_service.get_versions(dataset_id)
        
        return success_response({
            'versions': [v.to_dict() for v in versions]
        })
        
    except DatasetNotFoundError as e:
        return error_response(str(e), 404)
    except Exception as e:
        logger.error(f"Failed to get versions for dataset {dataset_id}: {e}")
        return error_response(f"获取版本列表失败: {str(e)}", 500)


@dataset_bp.route('/<dataset_id>/versions', methods=['POST'])
@jwt_required()
def create_dataset_version(dataset_id):
    """创建数据集新版本
    
    为数据集创建新版本。
    
    路径参数:
        dataset_id: string - 数据集ID
        
    请求体:
        {
            "version": "string",        # 必填，版本号
            "description": "string",    # 可选，版本描述
            "changes": {}               # 可选，变更内容
        }
        
    响应:
        201 Created: 新版本信息
        
    示例:
        curl -X POST /api/v1/datasets/550e8400-e29b-41d4-a716-446655440000/versions \\
            -H "Authorization: Bearer <token>" \\
            -H "Content-Type: application/json" \\
            -d '{"version": "2.0", "description": "添加新数据"}'
    """
    try:
        user_id = get_jwt_identity()
        
        # 权限检查
        dataset = dataset_service.get_dataset(dataset_id)
        has_permission, error = check_dataset_permission(dataset, user_id)
        if not has_permission:
            return error
        
        data = request.get_json()
        if not data:
            return error_response("请求数据不能为空", 400)
            
        version = data.get('version')
        if not version:
            return error_response("version不能为空", 400)
            
        description = data.get('description')
        changes = data.get('changes')
        
        new_version = dataset_service.create_version(
            dataset_id=dataset_id,
            version=version,
            description=description,
            created_by=user_id,
            changes=changes
        )
        
        return success_response(new_version.to_dict(), "版本创建成功", 201)
        
    except DatasetNotFoundError as e:
        return error_response(str(e), 404)
    except Exception as e:
        logger.error(f"Failed to create version for dataset {dataset_id}: {e}")
        return error_response(f"创建版本失败: {str(e)}", 500)


# ============================================================================
# 访问日志接口
# ============================================================================

@dataset_bp.route('/<dataset_id>/access-logs', methods=['GET'])
@jwt_required()
def get_dataset_access_logs(dataset_id):
    """获取数据集访问日志
    
    获取数据集的访问历史记录。
    
    路径参数:
        dataset_id: string - 数据集ID
        
    查询参数:
        limit: int - 返回数量限制，默认100
        offset: int - 偏移量，默认0
        
    响应:
        200 OK:
            {
                "success": true,
                "data": {
                    "logs": [
                        {
                            "log_id": "uuid",
                            "user_id": "string",
                            "action": "string",
                            "details": {},
                            "ip_address": "string",
                            "timestamp": "ISO8601"
                        }
                    ]
                }
            }
            
    示例:
        curl -X GET "/api/v1/datasets/550e8400-e29b-41d4-a716-446655440000/access-logs?limit=50" \\
            -H "Authorization: Bearer <token>"
    """
    try:
        user_id = get_jwt_identity()
        
        # 权限检查
        dataset = dataset_service.get_dataset(dataset_id)
        has_permission, error = check_dataset_permission(dataset, user_id)
        if not has_permission:
            return error
        
        limit = request.args.get('limit', 100, type=int)
        offset = request.args.get('offset', 0, type=int)
        
        logs = dataset_service.get_access_logs(
            dataset_id=dataset_id,
            limit=limit,
            offset=offset
        )
        
        return success_response({'logs': logs})
        
    except DatasetNotFoundError as e:
        return error_response(str(e), 404)
    except Exception as e:
        logger.error(f"Failed to get access logs for dataset {dataset_id}: {e}")
        return error_response(f"获取访问日志失败: {str(e)}", 500)

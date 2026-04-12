"""数据发现与接入API

提供数据发现与接入相关的API接口。
"""

import sys
import os
from typing import Dict, Any, Optional
from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
import logging

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from backend.core.exceptions import ValidationError
from backend.services.data_discovery_service import DataDiscoveryService
from backend.repositories.dataset_repository import DatasetRepository
from backend.modules.dataset.dataset_exceptions import (
    DatasetNotFoundError,
    DatasetValidationError,
    DataDiscoveryError,
    DataSourceNotFoundError,
    DiscoveryNotFoundError,
    DataIngestionError,
    SchemaInferenceError,
    DataTransformationError,
    SyncConfigurationError,
)
from backend.utils.response import success_response, error_response, paginated_response

logger = logging.getLogger(__name__)

# 创建蓝图
data_discovery_bp = Blueprint('data_discovery', __name__, url_prefix='/api/v1/datasets')

# 初始化服务
dataset_repository = DatasetRepository()
discovery_service = DataDiscoveryService(dataset_repository)


# ============================================================================
# 数据源管理API
# ============================================================================

@data_discovery_bp.route('/sources', methods=['GET'])
@jwt_required()
def list_data_sources():
    """获取数据源列表
    
    Query Parameters:
        - tenant_id: 租户ID（可选）
        - page: 页码（默认1）
        - page_size: 每页数量（默认20，最大100）
    
    Returns:
        数据源列表
    """
    try:
        user_id = get_jwt_identity()
        
        # 获取查询参数
        tenant_id = request.args.get('tenant_id')
        page = int(request.args.get('page', 1))
        page_size = min(int(request.args.get('page_size', 20)), 100)
        offset = (page - 1) * page_size
        
        # 获取数据源列表
        sources = discovery_service.list_data_sources(
            user_id=user_id,
            tenant_id=tenant_id,
            limit=page_size,
            offset=offset
        )
        
        return success_response(
            data=sources,
            message="获取数据源列表成功"
        )
    except Exception as e:
        logger.error(f"Error listing data sources: {e}")
        return error_response(
            message=f"获取数据源列表时发生错误: {str(e)}",
            code=500,
            error_type="LIST_SOURCES_ERROR"
        ), 500


@data_discovery_bp.route('/sources/<source_id>', methods=['DELETE'])
@jwt_required()
def delete_data_source(source_id: str):
    """删除数据源
    
    Args:
        source_id: 数据源ID
        
    Returns:
        删除结果
    """
    try:
        user_id = get_jwt_identity()
        
        result = discovery_service.delete_data_source(source_id, user_id)
        
        if result:
            return success_response(
                data={"deleted": True},
                message="数据源删除成功"
            )
        else:
            return error_response(
                message="删除数据源失败",
                code=400,
                error_type="DELETE_SOURCE_ERROR"
            ), 400
            
    except DataSourceNotFoundError as e:
        return error_response(
            message=str(e),
            code=404,
            error_type="SOURCE_NOT_FOUND"
        ), 404
    except Exception as e:
        logger.error(f"Error deleting data source: {e}")
        return error_response(
            message=f"删除数据源时发生错误: {str(e)}",
            code=500,
            error_type="DELETE_SOURCE_ERROR"
        ), 500


@data_discovery_bp.route('/sources/scan', methods=['POST'])
@jwt_required()
def scan_data_sources():
    """扫描数据源
    
    Request Body:
        - sources: 要扫描的数据源配置列表
        - parallel_scan: 是否并行扫描（默认true）
        - include_preview: 是否包含数据预览（默认true）
        - preview_rows: 预览行数（默认5）
        - tenant_id: 租户ID（可选）
        
    Returns:
        扫描结果
    """
    try:
        user_id = get_jwt_identity()
        scan_config = request.get_json() or {}
        
        logger.info(f"User {user_id} scanning data sources")
        
        sources = discovery_service.scan_data_sources(user_id, scan_config)
        
        return success_response(
            data={
                "sources": sources,
                "total_scanned": len(sources),
                "success_count": sum(1 for s in sources if s.get('status') == 'discovered'),
                "failed_count": sum(1 for s in sources if s.get('status') == 'failed'),
            },
            message="数据源扫描完成"
        )
    except DataDiscoveryError as e:
        logger.error(f"Discovery error during scan: {e}")
        return error_response(
            message=str(e),
            code=400,
            error_type="DISCOVERY_ERROR"
        ), 400
    except Exception as e:
        logger.error(f"Error scanning data sources: {e}")
        return error_response(
            message=f"扫描数据源时发生错误: {str(e)}",
            code=500,
            error_type="SCAN_ERROR"
        ), 500


# ============================================================================
# 数据集发现API
# ============================================================================

@data_discovery_bp.route('/discover', methods=['POST'])
@jwt_required()
def discover_datasets():
    """发现数据集
    
    Request Body:
        - source_ids: 指定数据源ID列表（可选，不指定则扫描所有）
        - auto_detect_format: 自动检测数据格式（默认true）
        - auto_detect_schema: 自动检测数据模式（默认true）
        - sample_size: 采样大小（默认1000）
        - include_statistics: 是否包含统计信息（默认true）
        - tenant_id: 租户ID（可选）
        
    Returns:
        发现的数据集列表
    """
    try:
        user_id = get_jwt_identity()
        discovery_config = request.get_json() or {}
        
        logger.info(f"User {user_id} discovering datasets")
        
        datasets = discovery_service.discover_datasets(user_id, discovery_config)
        
        return success_response(
            data={
                "datasets": datasets,
                "total_discovered": len(datasets),
            },
            message="数据集发现完成"
        )
    except DataDiscoveryError as e:
        logger.error(f"Discovery error: {e}")
        return error_response(
            message=str(e),
            code=400,
            error_type="DISCOVERY_ERROR"
        ), 400
    except Exception as e:
        logger.error(f"Error discovering datasets: {e}")
        return error_response(
            message=f"发现数据集时发生错误: {str(e)}",
            code=500,
            error_type="DISCOVERY_ERROR"
        ), 500


@data_discovery_bp.route('/discoveries', methods=['GET'])
@jwt_required()
def list_discoveries():
    """获取发现记录列表
    
    Query Parameters:
        - page: 页码（默认1）
        - page_size: 每页数量（默认20）
        - status: 状态过滤（可多选，逗号分隔）
        - source_type: 数据源类型过滤（可多选，逗号分隔）
        - tenant_id: 租户ID（可选）
        
    Returns:
        发现记录列表
    """
    try:
        user_id = get_jwt_identity()
        
        # 获取查询参数
        page = int(request.args.get('page', 1))
        page_size = min(int(request.args.get('page_size', 20)), 100)
        status_filter = request.args.get('status', '').split(',') if request.args.get('status') else None
        source_type_filter = request.args.get('source_type', '').split(',') if request.args.get('source_type') else None
        
        # 过滤空字符串
        if status_filter:
            status_filter = [s for s in status_filter if s]
        if source_type_filter:
            source_type_filter = [s for s in source_type_filter if s]
        
        offset = (page - 1) * page_size
        
        # 获取发现记录
        records = discovery_service.list_discoveries(
            user_id=user_id,
            limit=page_size,
            offset=offset
        )
        
        return success_response(
            data={
                "records": records,
                "page": page,
                "page_size": page_size,
            },
            message="获取发现记录成功"
        )
    except DatasetValidationError as e:
        return error_response(
            message=str(e),
            code=400,
            error_type="VALIDATION_ERROR"
        ), 400
    except Exception as e:
        logger.error(f"Error listing discoveries: {e}")
        return error_response(
            message=f"获取发现记录时发生错误: {str(e)}",
            code=500,
            error_type="LIST_ERROR"
        ), 500


@data_discovery_bp.route('/discoveries/<record_id>', methods=['GET'])
@jwt_required()
def get_discovery_details(record_id: str):
    """获取发现记录详情
    
    Args:
        record_id: 发现记录ID
        
    Returns:
        发现记录详情
    """
    try:
        user_id = get_jwt_identity()
        
        details = discovery_service.get_discovery_details(record_id, user_id)
        
        return success_response(
            data=details,
            message="获取发现记录详情成功"
        )
    except DiscoveryNotFoundError as e:
        return error_response(
            message=str(e),
            code=404,
            error_type="DISCOVERY_NOT_FOUND"
        ), 404
    except Exception as e:
        logger.error(f"Error getting discovery details: {e}")
        return error_response(
            message=f"获取发现记录详情时发生错误: {str(e)}",
            code=500,
            error_type="GET_DETAILS_ERROR"
        ), 500


# ============================================================================
# 数据集接入API
# ============================================================================

@data_discovery_bp.route('/ingest', methods=['POST'])
@jwt_required()
def auto_ingest_dataset():
    """自动接入数据集
    
    Request Body:
        - discovery_id: 发现ID（从发现结果接入）
        - source_config: 数据源配置（直接接入，与discovery_id二选一）
        - name: 数据集名称（可选）
        - description: 数据集描述（可选）
        - dataset_type: 数据集类型（默认generic）
        - storage_path: 存储路径（可选）
        - enable_sync: 是否启用同步（默认false）
        - sync_frequency: 同步频率（默认daily）
        - tenant_id: 租户ID（可选）
        
    Returns:
        接入的数据集信息
    """
    try:
        user_id = get_jwt_identity()
        source_info = request.get_json() or {}
        
        logger.info(f"User {user_id} ingesting dataset")
        
        # 验证请求
        if not source_info.get('discovery_id') and not source_info.get('source_config'):
            return error_response(
                message="必须提供 discovery_id 或 source_config",
                code=400,
                error_type="VALIDATION_ERROR"
            ), 400
        
        dataset = discovery_service.auto_ingest_dataset(user_id, source_info)
        
        return success_response(
            data=dataset.to_dict(),
            message="数据集接入完成"
        )
    except DiscoveryNotFoundError as e:
        return error_response(
            message=str(e),
            code=404,
            error_type="DISCOVERY_NOT_FOUND"
        ), 404
    except DataIngestionError as e:
        logger.error(f"Ingestion error: {e}")
        return error_response(
            message=str(e),
            code=400,
            error_type="INGESTION_ERROR"
        ), 400
    except Exception as e:
        logger.error(f"Error ingesting dataset: {e}")
        return error_response(
            message=f"接入数据集时发生错误: {str(e)}",
            code=500,
            error_type="INGESTION_ERROR"
        ), 500


@data_discovery_bp.route('/ingest/batch', methods=['POST'])
@jwt_required()
def batch_ingest_datasets():
    """批量接入数据集
    
    Request Body:
        - discovery_ids: 发现ID列表
        - ingest_config: 通用接入配置
        - tenant_id: 租户ID（可选）
        
    Returns:
        批量接入结果
    """
    try:
        user_id = get_jwt_identity()
        batch_config = request.get_json() or {}
        
        discovery_ids = batch_config.get('discovery_ids', [])
        ingest_config = batch_config.get('ingest_config', {})
        tenant_id = batch_config.get('tenant_id')
        
        if not discovery_ids:
            return error_response(
                message="必须提供 discovery_ids",
                code=400,
                error_type="VALIDATION_ERROR"
            ), 400
        
        logger.info(f"User {user_id} batch ingesting {len(discovery_ids)} datasets")
        
        results = {
            "success": [],
            "failed": [],
            "total": len(discovery_ids),
        }
        
        for discovery_id in discovery_ids:
            try:
                source_info = {
                    'discovery_id': discovery_id,
                    'tenant_id': tenant_id,
                    **ingest_config
                }
                dataset = discovery_service.auto_ingest_dataset(user_id, source_info)
                results["success"].append({
                    "discovery_id": discovery_id,
                    "dataset_id": dataset.dataset_id,
                    "dataset_name": dataset.name,
                })
            except Exception as e:
                results["failed"].append({
                    "discovery_id": discovery_id,
                    "error": str(e),
                })
        
        results["success_count"] = len(results["success"])
        results["failed_count"] = len(results["failed"])
        
        return success_response(
            data=results,
            message=f"批量接入完成: {results['success_count']} 成功, {results['failed_count']} 失败"
        )
    except Exception as e:
        logger.error(f"Error batch ingesting datasets: {e}")
        return error_response(
            message=f"批量接入数据集时发生错误: {str(e)}",
            code=500,
            error_type="BATCH_INGESTION_ERROR"
        ), 500


# ============================================================================
# 数据模式推断API
# ============================================================================

@data_discovery_bp.route('/<dataset_id>/schema/infer', methods=['POST'])
@jwt_required()
def infer_schema(dataset_id: str):
    """推断数据模式
    
    Args:
        dataset_id: 数据集ID
        
    Request Body:
        - sample_size: 采样大小（默认1000）
        - detect_nested: 是否检测嵌套结构（默认true）
        - infer_constraints: 是否推断约束（默认true）
        
    Returns:
        推断的数据模式
    """
    try:
        user_id = get_jwt_identity()
        infer_config = request.get_json() or {}
        
        logger.info(f"User {user_id} inferring schema for dataset {dataset_id}")
        
        schema = discovery_service.infer_schema(dataset_id)
        
        return success_response(
            data=schema,
            message="数据模式推断完成"
        )
    except DatasetNotFoundError as e:
        return error_response(
            message=str(e),
            code=404,
            error_type="DATASET_NOT_FOUND"
        ), 404
    except SchemaInferenceError as e:
        logger.error(f"Schema inference error: {e}")
        return error_response(
            message=str(e),
            code=400,
            error_type="SCHEMA_INFERENCE_ERROR"
        ), 400
    except Exception as e:
        logger.error(f"Error inferring schema: {e}")
        return error_response(
            message=f"推断数据模式时发生错误: {str(e)}",
            code=500,
            error_type="SCHEMA_INFERENCE_ERROR"
        ), 500


@data_discovery_bp.route('/<dataset_id>/schema', methods=['GET'])
@jwt_required()
def get_dataset_schema(dataset_id: str):
    """获取数据集的已保存模式
    
    Args:
        dataset_id: 数据集ID
        
    Returns:
        数据集模式信息
    """
    try:
        user_id = get_jwt_identity()
        
        # 获取数据集
        dataset = dataset_repository.get_by_id(dataset_id)
        if not dataset:
            raise DatasetNotFoundError(f"数据集 {dataset_id} 不存在")
        
        # 获取已保存的模式
        config = getattr(dataset, 'config', {}) or {}
        schema = config.get('inferred_schema')
        
        if not schema:
            return error_response(
                message="数据集尚未推断模式，请先调用推断接口",
                code=404,
                error_type="SCHEMA_NOT_FOUND"
            ), 404
        
        return success_response(
            data=schema,
            message="获取数据模式成功"
        )
    except DatasetNotFoundError as e:
        return error_response(
            message=str(e),
            code=404,
            error_type="DATASET_NOT_FOUND"
        ), 404
    except Exception as e:
        logger.error(f"Error getting schema: {e}")
        return error_response(
            message=f"获取数据模式时发生错误: {str(e)}",
            code=500,
            error_type="GET_SCHEMA_ERROR"
        ), 500


# ============================================================================
# 数据转换API
# ============================================================================

@data_discovery_bp.route('/<dataset_id>/transform/auto', methods=['POST'])
@jwt_required()
def auto_transform(dataset_id: str):
    """自动转换数据
    
    Args:
        dataset_id: 数据集ID
        
    Request Body:
        - operations: 转换操作列表
        - auto_normalize: 是否自动规范化（默认true）
        - auto_handle_missing: 是否自动处理缺失值（默认true）
        - create_new_dataset: 是否创建新数据集（默认false）
        - output_format: 输出格式（可选）
        
    Returns:
        转换后的数据集信息
    """
    try:
        user_id = get_jwt_identity()
        transform_config = request.get_json() or {}
        
        logger.info(f"User {user_id} transforming dataset {dataset_id}")
        
        dataset = discovery_service.auto_transform(dataset_id, transform_config)
        
        return success_response(
            data=dataset.to_dict(),
            message="数据转换完成"
        )
    except DatasetNotFoundError as e:
        return error_response(
            message=str(e),
            code=404,
            error_type="DATASET_NOT_FOUND"
        ), 404
    except DataTransformationError as e:
        logger.error(f"Transformation error: {e}")
        return error_response(
            message=str(e),
            code=400,
            error_type="TRANSFORMATION_ERROR"
        ), 400
    except Exception as e:
        logger.error(f"Error transforming data: {e}")
        return error_response(
            message=f"转换数据时发生错误: {str(e)}",
            code=500,
            error_type="TRANSFORMATION_ERROR"
        ), 500


# ============================================================================
# 增量同步API
# ============================================================================

@data_discovery_bp.route('/<dataset_id>/sync/setup', methods=['POST'])
@jwt_required()
def setup_incremental_sync(dataset_id: str):
    """设置增量同步
    
    Args:
        dataset_id: 数据集ID
        
    Request Body:
        - sync_enabled: 是否启用同步（默认true）
        - frequency: 同步频率（realtime/hourly/daily/weekly/monthly）
        - incremental_column: 增量列名
        - incremental_method: 增量方法（timestamp/id/hash）
        - cron_expression: Cron表达式（自定义频率时使用）
        - timezone: 时区（默认UTC）
        - conflict_resolution: 冲突处理方式（update/skip/error）
        
    Returns:
        配置后的数据集信息
    """
    try:
        user_id = get_jwt_identity()
        sync_config = request.get_json() or {}
        
        logger.info(f"User {user_id} setting up sync for dataset {dataset_id}")
        
        dataset = discovery_service.setup_incremental_sync(dataset_id, sync_config)
        
        return success_response(
            data=dataset.to_dict(),
            message="增量同步设置完成"
        )
    except DatasetNotFoundError as e:
        return error_response(
            message=str(e),
            code=404,
            error_type="DATASET_NOT_FOUND"
        ), 404
    except SyncConfigurationError as e:
        logger.error(f"Sync configuration error: {e}")
        return error_response(
            message=str(e),
            code=400,
            error_type="SYNC_SETUP_ERROR"
        ), 400
    except Exception as e:
        logger.error(f"Error setting up sync: {e}")
        return error_response(
            message=f"设置增量同步时发生错误: {str(e)}",
            code=500,
            error_type="SYNC_SETUP_ERROR"
        ), 500


@data_discovery_bp.route('/<dataset_id>/sync/status', methods=['GET'])
@jwt_required()
def get_sync_status(dataset_id: str):
    """获取同步状态
    
    Args:
        dataset_id: 数据集ID
        
    Returns:
        同步状态信息
    """
    try:
        user_id = get_jwt_identity()
        
        status = discovery_service.get_sync_status(dataset_id)
        
        return success_response(
            data=status,
            message="获取同步状态成功"
        )
    except DatasetNotFoundError as e:
        return error_response(
            message=str(e),
            code=404,
            error_type="DATASET_NOT_FOUND"
        ), 404
    except Exception as e:
        logger.error(f"Error getting sync status: {e}")
        return error_response(
            message=f"获取同步状态时发生错误: {str(e)}",
            code=500,
            error_type="GET_SYNC_STATUS_ERROR"
        ), 500


@data_discovery_bp.route('/<dataset_id>/sync/disable', methods=['POST'])
@jwt_required()
def disable_sync(dataset_id: str):
    """禁用同步
    
    Args:
        dataset_id: 数据集ID
        
    Returns:
        操作结果
    """
    try:
        user_id = get_jwt_identity()
        
        logger.info(f"User {user_id} disabling sync for dataset {dataset_id}")
        
        # 设置同步为禁用
        dataset = discovery_service.setup_incremental_sync(
            dataset_id, 
            {'sync_enabled': False}
        )
        
        return success_response(
            data={"sync_enabled": False},
            message="同步已禁用"
        )
    except DatasetNotFoundError as e:
        return error_response(
            message=str(e),
            code=404,
            error_type="DATASET_NOT_FOUND"
        ), 404
    except Exception as e:
        logger.error(f"Error disabling sync: {e}")
        return error_response(
            message=f"禁用同步时发生错误: {str(e)}",
            code=500,
            error_type="DISABLE_SYNC_ERROR"
        ), 500


# ============================================================================
# 数据发现分析API（接口兼容）
# ============================================================================

@data_discovery_bp.route('/<dataset_id>/discover', methods=['POST'])
@jwt_required()
def discover_dataset(dataset_id: str):
    """对已存在的数据集执行发现分析
    
    Args:
        dataset_id: 数据集ID
        
    Returns:
        发现分析结果
    """
    try:
        user_id = get_jwt_identity()
        
        logger.info(f"User {user_id} running discovery on dataset {dataset_id}")
        
        result = discovery_service.discover(dataset_id)
        
        return success_response(
            data=result,
            message="数据发现分析完成"
        )
    except DatasetNotFoundError as e:
        return error_response(
            message=str(e),
            code=404,
            error_type="DATASET_NOT_FOUND"
        ), 404
    except Exception as e:
        logger.error(f"Error discovering dataset: {e}")
        return error_response(
            message=f"数据发现分析时发生错误: {str(e)}",
            code=500,
            error_type="DISCOVERY_ERROR"
        ), 500


# ============================================================================
# 健康检查和统计API
# ============================================================================

@data_discovery_bp.route('/health', methods=['GET'])
def health_check():
    """健康检查
    
    Returns:
        服务健康状态
    """
    return success_response(
        data={
            "status": "healthy",
            "service": "data_discovery",
            "version": "1.0.0",
        },
        message="服务运行正常"
    )


@data_discovery_bp.route('/stats', methods=['GET'])
@jwt_required()
def get_discovery_stats():
    """获取用户的数据发现统计信息
    
    Returns:
        统计信息
    """
    try:
        user_id = get_jwt_identity()
        tenant_id = request.args.get('tenant_id')
        
        # 获取统计数据
        sources = discovery_service.list_data_sources(user_id, tenant_id)
        discoveries = discovery_service.list_discoveries(user_id)
        
        stats = {
            "total_sources": len(sources),
            "active_sources": sum(1 for s in sources if s.get('status') == 'active'),
            "total_discoveries": len(discoveries),
            "pending_discoveries": sum(1 for d in discoveries if d.get('status') == 'pending'),
            "completed_discoveries": sum(1 for d in discoveries if d.get('status') == 'discovered'),
            "total_datasets_discovered": sum(d.get('datasets_discovered', 0) for d in discoveries),
            "total_datasets_ingested": sum(d.get('datasets_ingested', 0) for d in discoveries),
        }
        
        return success_response(
            data=stats,
            message="获取统计信息成功"
        )
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        return error_response(
            message=f"获取统计信息时发生错误: {str(e)}",
            code=500,
            error_type="GET_STATS_ERROR"
        ), 500

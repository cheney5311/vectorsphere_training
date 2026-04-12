"""数据预处理API

提供数据预处理相关的API接口。
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
from backend.services.data_preprocessing_service import DataPreprocessingService
from backend.repositories.dataset_repository import DatasetRepository
from backend.modules.dataset.dataset_exceptions import (
    DatasetNotFoundError,
    DatasetValidationError,
    DataPreprocessingError,
    PreprocessingTaskNotFoundError,
    PreprocessingOperationError,
    PreprocessingConfigError,
    FeatureEngineeringError,
    DataAugmentationError,
    DataSplitError,
    DataSplitRatioError,
    InsufficientDataError,
)
from backend.utils.response import success_response, error_response, paginated_response

logger = logging.getLogger(__name__)

# 创建蓝图
data_preprocessing_bp = Blueprint('data_preprocessing', __name__, url_prefix='/api/v1/datasets')

# 初始化服务
dataset_repository = DatasetRepository()
preprocessing_service = DataPreprocessingService(dataset_repository)


# ============================================================================
# 数据预处理API
# ============================================================================

@data_preprocessing_bp.route('/<dataset_id>/preprocess', methods=['POST'])
@jwt_required()
def preprocess_dataset(dataset_id: str):
    """预处理数据集
    
    Args:
        dataset_id: 数据集ID
        
    Request Body:
        - pipeline_config: 流水线配置（高级模式）
        - normalize: 是否标准化
        - normalize_method: 标准化方法
        - normalize_columns: 标准化的列
        - tokenize: 是否分词
        - tokenize_columns: 分词的列
        - language: 语言
        - filter_invalid: 是否过滤无效数据
        - filter_conditions: 过滤条件
        - remove_duplicates: 是否去重
        - handle_missing: 缺失值处理配置
        - handle_outliers: 异常值处理配置
        
    Returns:
        预处理后的数据集信息
    """
    try:
        user_id = get_jwt_identity()
        preprocessing_config = request.get_json() or {}
        
        logger.info(f"User {user_id} preprocessing dataset {dataset_id}")
        
        dataset = preprocessing_service.preprocess_dataset(dataset_id, preprocessing_config)
        
        return success_response(
            data=dataset.to_dict(),
            message="数据预处理完成"
        )
    except DatasetNotFoundError as e:
        return error_response(
            message=str(e),
            code=404,
            error_type="DATASET_NOT_FOUND"
        ), 404
    except PreprocessingConfigError as e:
        return error_response(
            message=str(e),
            code=400,
            error_type="CONFIG_ERROR"
        ), 400
    except PreprocessingOperationError as e:
        logger.error(f"Preprocessing operation error: {e}")
        return error_response(
            message=str(e),
            code=400,
            error_type="OPERATION_ERROR"
        ), 400
    except Exception as e:
        logger.error(f"Error preprocessing dataset: {e}")
        return error_response(
            message=f"预处理数据集时发生错误: {str(e)}",
            code=500,
            error_type="PREPROCESSING_ERROR"
        ), 500


@data_preprocessing_bp.route('/<dataset_id>/preprocess/pipeline', methods=['POST'])
@jwt_required()
def preprocess_with_pipeline(dataset_id: str):
    """使用流水线预处理数据集
    
    Args:
        dataset_id: 数据集ID
        
    Request Body:
        - pipeline_id: 流水线ID
        
    Returns:
        预处理后的数据集信息
    """
    try:
        user_id = get_jwt_identity()
        config = request.get_json() or {}
        pipeline_id = config.get('pipeline_id')
        
        if not pipeline_id:
            return error_response(
                message="必须提供 pipeline_id",
                code=400,
                error_type="VALIDATION_ERROR"
            ), 400
        
        logger.info(f"User {user_id} preprocessing dataset {dataset_id} with pipeline {pipeline_id}")
        
        dataset = preprocessing_service.execute_pipeline(dataset_id, pipeline_id)
        
        return success_response(
            data=dataset.to_dict(),
            message="流水线预处理完成"
        )
    except DatasetNotFoundError as e:
        return error_response(
            message=str(e),
            code=404,
            error_type="DATASET_NOT_FOUND"
        ), 404
    except DataPreprocessingError as e:
        return error_response(
            message=str(e),
            code=400,
            error_type="PREPROCESSING_ERROR"
        ), 400
    except Exception as e:
        logger.error(f"Error executing pipeline: {e}")
        return error_response(
            message=f"执行流水线时发生错误: {str(e)}",
            code=500,
            error_type="PIPELINE_ERROR"
        ), 500


# ============================================================================
# 特征工程API
# ============================================================================

@data_preprocessing_bp.route('/<dataset_id>/features/engineer', methods=['POST'])
@jwt_required()
def perform_feature_engineering(dataset_id: str):
    """执行特征工程
    
    Args:
        dataset_id: 数据集ID
        
    Request Body:
        - create_features: 要创建的特征列表
          - name: 特征名称
          - expression: 计算表达式
          - description: 特征描述
        - feature_selection: 特征选择配置
          - method: 选择方法
          - threshold: 阈值
          - exclude_columns: 排除的列
        - feature_transform: 特征转换配置列表
          - columns: 目标列
          - transform_type: 转换类型
        - dimension_reduction: 降维配置
          - method: 降维方法
          - n_components: 目标维度
        - encoding: 编码配置列表
          - columns: 目标列
          - method: 编码方法
          
    Returns:
        特征工程后的数据集信息
    """
    try:
        user_id = get_jwt_identity()
        features_config = request.get_json() or {}
        
        logger.info(f"User {user_id} performing feature engineering on dataset {dataset_id}")
        
        dataset = preprocessing_service.perform_feature_engineering(dataset_id, features_config)
        
        return success_response(
            data=dataset.to_dict(),
            message="特征工程完成"
        )
    except DatasetNotFoundError as e:
        return error_response(
            message=str(e),
            code=404,
            error_type="DATASET_NOT_FOUND"
        ), 404
    except FeatureEngineeringError as e:
        logger.error(f"Feature engineering error: {e}")
        return error_response(
            message=str(e),
            code=400,
            error_type="FEATURE_ENGINEERING_ERROR"
        ), 400
    except Exception as e:
        logger.error(f"Error performing feature engineering: {e}")
        return error_response(
            message=f"执行特征工程时发生错误: {str(e)}",
            code=500,
            error_type="FEATURE_ENGINEERING_ERROR"
        ), 500


@data_preprocessing_bp.route('/<dataset_id>/features', methods=['GET'])
@jwt_required()
def list_dataset_features(dataset_id: str):
    """获取数据集的特征列表
    
    Args:
        dataset_id: 数据集ID
        
    Query Parameters:
        - feature_type: 特征类型过滤
        
    Returns:
        特征列表
    """
    try:
        user_id = get_jwt_identity()
        feature_type = request.args.get('feature_type')
        
        # 从特征存储获取特征
        from backend.repositories.data_preprocessing_repository import get_preprocessing_repository_manager
        repo_manager = get_preprocessing_repository_manager()
        
        feature_type_filter = [feature_type] if feature_type else None
        features = repo_manager.feature_repo.get_by_dataset(
            dataset_id=dataset_id,
            feature_type_filter=feature_type_filter
        )
        
        return success_response(
            data=[f.to_dict() for f in features],
            message="获取特征列表成功"
        )
    except Exception as e:
        logger.error(f"Error listing features: {e}")
        return error_response(
            message=f"获取特征列表时发生错误: {str(e)}",
            code=500,
            error_type="LIST_FEATURES_ERROR"
        ), 500


# ============================================================================
# 数据增强API
# ============================================================================

@data_preprocessing_bp.route('/<dataset_id>/augment', methods=['POST'])
@jwt_required()
def perform_data_augmentation(dataset_id: str):
    """执行数据增强
    
    Args:
        dataset_id: 数据集ID
        
    Request Body:
        - augmentation_type: 增强类型（text/image/tabular）
        - text_config: 文本增强配置
          - methods: 增强方法列表
          - columns: 目标列
          - augment_ratio: 增强比例
          - num_augment: 每条增强次数
        - image_config: 图像增强配置
          - methods: 增强方法列表
          - columns: 目标列
          - augment_ratio: 增强比例
        - sampling_config: 采样配置
          - method: 采样方法（oversample/undersample/smote）
          - target_column: 目标列
          - sampling_strategy: 采样策略
        - target_size: 目标数据量
        - keep_original: 是否保留原始数据
        
    Returns:
        数据增强后的数据集信息
    """
    try:
        user_id = get_jwt_identity()
        augmentation_config = request.get_json() or {}
        
        logger.info(f"User {user_id} performing data augmentation on dataset {dataset_id}")
        
        dataset = preprocessing_service.perform_data_augmentation(dataset_id, augmentation_config)
        
        return success_response(
            data=dataset.to_dict(),
            message="数据增强完成"
        )
    except DatasetNotFoundError as e:
        return error_response(
            message=str(e),
            code=404,
            error_type="DATASET_NOT_FOUND"
        ), 404
    except DataAugmentationError as e:
        logger.error(f"Data augmentation error: {e}")
        return error_response(
            message=str(e),
            code=400,
            error_type="AUGMENTATION_ERROR"
        ), 400
    except Exception as e:
        logger.error(f"Error performing data augmentation: {e}")
        return error_response(
            message=f"执行数据增强时发生错误: {str(e)}",
            code=500,
            error_type="AUGMENTATION_ERROR"
        ), 500


# ============================================================================
# 数据分割API
# ============================================================================

@data_preprocessing_bp.route('/<dataset_id>/split', methods=['POST'])
@jwt_required()
def split_dataset(dataset_id: str):
    """分割数据集
    
    Args:
        dataset_id: 数据集ID
        
    Request Body:
        - train_ratio: 训练集比例（默认0.8）
        - val_ratio: 验证集比例（默认0.1）
        - test_ratio: 测试集比例（默认0.1）
        - stratify_column: 分层采样目标列
        - shuffle: 是否打乱数据
        - random_state: 随机种子
        - cross_validation: 交叉验证配置
          - n_folds: 折数
          - stratify_column: 分层目标列
        - output_format: 输出格式
        - create_new_datasets: 是否创建新数据集
        
    Returns:
        分割后的数据集信息
    """
    try:
        user_id = get_jwt_identity()
        split_config = request.get_json() or {}
        
        logger.info(f"User {user_id} splitting dataset {dataset_id}")
        
        split_result = preprocessing_service.split_dataset(dataset_id, split_config)
        
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
    except DataSplitRatioError as e:
        return error_response(
            message=str(e),
            code=400,
            error_type="SPLIT_RATIO_ERROR"
        ), 400
    except InsufficientDataError as e:
        return error_response(
            message=str(e),
            code=400,
            error_type="INSUFFICIENT_DATA"
        ), 400
    except DataSplitError as e:
        logger.error(f"Data split error: {e}")
        return error_response(
            message=str(e),
            code=400,
            error_type="SPLIT_ERROR"
        ), 400
    except Exception as e:
        logger.error(f"Error splitting dataset: {e}")
        return error_response(
            message=f"分割数据集时发生错误: {str(e)}",
            code=500,
            error_type="SPLIT_ERROR"
        ), 500


# ============================================================================
# 预处理任务API
# ============================================================================

@data_preprocessing_bp.route('/<dataset_id>/tasks', methods=['GET'])
@jwt_required()
def list_preprocessing_tasks(dataset_id: str):
    """获取数据集的预处理任务列表
    
    Args:
        dataset_id: 数据集ID
        
    Query Parameters:
        - status: 状态过滤（逗号分隔）
        - task_type: 任务类型过滤（逗号分隔）
        - page: 页码
        - page_size: 每页大小
        
    Returns:
        任务列表
    """
    try:
        user_id = get_jwt_identity()
        
        status_filter = request.args.get('status', '').split(',') if request.args.get('status') else None
        task_type_filter = request.args.get('task_type', '').split(',') if request.args.get('task_type') else None
        page = int(request.args.get('page', 1))
        page_size = min(int(request.args.get('page_size', 20)), 100)
        
        # 过滤空字符串
        if status_filter:
            status_filter = [s for s in status_filter if s]
        if task_type_filter:
            task_type_filter = [t for t in task_type_filter if t]
        
        result = preprocessing_service.list_tasks(
            dataset_id=dataset_id,
            status_filter=status_filter,
            task_type_filter=task_type_filter,
            page=page,
            page_size=page_size
        )
        
        return success_response(
            data=result,
            message="获取任务列表成功"
        )
    except Exception as e:
        logger.error(f"Error listing tasks: {e}")
        return error_response(
            message=f"获取任务列表时发生错误: {str(e)}",
            code=500,
            error_type="LIST_TASKS_ERROR"
        ), 500


@data_preprocessing_bp.route('/tasks/<task_id>', methods=['GET'])
@jwt_required()
def get_preprocessing_task(task_id: str):
    """获取预处理任务详情
    
    Args:
        task_id: 任务ID
        
    Returns:
        任务详情
    """
    try:
        user_id = get_jwt_identity()
        
        task = preprocessing_service.get_task(task_id)
        
        return success_response(
            data=task,
            message="获取任务详情成功"
        )
    except PreprocessingTaskNotFoundError as e:
        return error_response(
            message=str(e),
            code=404,
            error_type="TASK_NOT_FOUND"
        ), 404
    except Exception as e:
        logger.error(f"Error getting task: {e}")
        return error_response(
            message=f"获取任务详情时发生错误: {str(e)}",
            code=500,
            error_type="GET_TASK_ERROR"
        ), 500


@data_preprocessing_bp.route('/tasks/<task_id>/cancel', methods=['POST'])
@jwt_required()
def cancel_preprocessing_task(task_id: str):
    """取消预处理任务
    
    Args:
        task_id: 任务ID
        
    Returns:
        取消结果
    """
    try:
        user_id = get_jwt_identity()
        
        logger.info(f"User {user_id} cancelling task {task_id}")
        
        task = preprocessing_service.cancel_task(task_id)
        
        return success_response(
            data=task,
            message="任务取消成功"
        )
    except PreprocessingTaskNotFoundError as e:
        return error_response(
            message=str(e),
            code=404,
            error_type="TASK_NOT_FOUND"
        ), 404
    except DataPreprocessingError as e:
        return error_response(
            message=str(e),
            code=400,
            error_type="CANCEL_ERROR"
        ), 400
    except Exception as e:
        logger.error(f"Error cancelling task: {e}")
        return error_response(
            message=f"取消任务时发生错误: {str(e)}",
            code=500,
            error_type="CANCEL_TASK_ERROR"
        ), 500


# ============================================================================
# 预处理历史API
# ============================================================================

@data_preprocessing_bp.route('/<dataset_id>/history', methods=['GET'])
@jwt_required()
def get_preprocessing_history(dataset_id: str):
    """获取数据集的预处理历史记录
    
    Args:
        dataset_id: 数据集ID
        
    Query Parameters:
        - operation_type: 操作类型过滤（逗号分隔）
        - page: 页码
        - page_size: 每页大小
        
    Returns:
        历史记录列表
    """
    try:
        user_id = get_jwt_identity()
        
        operation_type_filter = request.args.get('operation_type', '').split(',') if request.args.get('operation_type') else None
        page = int(request.args.get('page', 1))
        page_size = min(int(request.args.get('page_size', 20)), 100)
        
        if operation_type_filter:
            operation_type_filter = [t for t in operation_type_filter if t]
        
        result = preprocessing_service.get_preprocessing_history(
            dataset_id=dataset_id,
            operation_type_filter=operation_type_filter,
            page=page,
            page_size=page_size
        )
        
        return success_response(
            data=result,
            message="获取历史记录成功"
        )
    except Exception as e:
        logger.error(f"Error getting history: {e}")
        return error_response(
            message=f"获取历史记录时发生错误: {str(e)}",
            code=500,
            error_type="GET_HISTORY_ERROR"
        ), 500


# ============================================================================
# 流水线管理API
# ============================================================================

@data_preprocessing_bp.route('/pipelines', methods=['POST'])
@jwt_required()
def create_pipeline():
    """创建预处理流水线
    
    Request Body:
        - name: 流水线名称
        - description: 描述
        - operations: 操作列表
        - is_template: 是否为模板
        - is_public: 是否公开
        
    Returns:
        创建的流水线信息
    """
    try:
        user_id = get_jwt_identity()
        config = request.get_json() or {}
        
        name = config.get('name')
        if not name:
            return error_response(
                message="必须提供流水线名称",
                code=400,
                error_type="VALIDATION_ERROR"
            ), 400
        
        operations = config.get('operations', [])
        if not operations:
            return error_response(
                message="必须提供至少一个操作",
                code=400,
                error_type="VALIDATION_ERROR"
            ), 400
        
        logger.info(f"User {user_id} creating pipeline: {name}")
        
        pipeline = preprocessing_service.create_pipeline(
            user_id=user_id,
            name=name,
            operations=operations,
            description=config.get('description'),
            is_template=config.get('is_template', False),
            is_public=config.get('is_public', False),
            tenant_id=config.get('tenant_id')
        )
        
        return success_response(
            data=pipeline,
            message="流水线创建成功"
        )
    except Exception as e:
        logger.error(f"Error creating pipeline: {e}")
        return error_response(
            message=f"创建流水线时发生错误: {str(e)}",
            code=500,
            error_type="CREATE_PIPELINE_ERROR"
        ), 500


@data_preprocessing_bp.route('/pipelines', methods=['GET'])
@jwt_required()
def list_pipelines():
    """获取流水线列表
    
    Query Parameters:
        - is_template: 是否为模板
        - page: 页码
        - page_size: 每页大小
        
    Returns:
        流水线列表
    """
    try:
        user_id = get_jwt_identity()
        
        is_template_str = request.args.get('is_template')
        is_template = None
        if is_template_str is not None:
            is_template = is_template_str.lower() == 'true'
        
        page = int(request.args.get('page', 1))
        page_size = min(int(request.args.get('page_size', 20)), 100)
        tenant_id = request.args.get('tenant_id')
        
        result = preprocessing_service.list_pipelines(
            user_id=user_id,
            tenant_id=tenant_id,
            is_template=is_template,
            page=page,
            page_size=page_size
        )
        
        return success_response(
            data=result,
            message="获取流水线列表成功"
        )
    except Exception as e:
        logger.error(f"Error listing pipelines: {e}")
        return error_response(
            message=f"获取流水线列表时发生错误: {str(e)}",
            code=500,
            error_type="LIST_PIPELINES_ERROR"
        ), 500


@data_preprocessing_bp.route('/pipelines/<pipeline_id>', methods=['GET'])
@jwt_required()
def get_pipeline(pipeline_id: str):
    """获取流水线详情
    
    Args:
        pipeline_id: 流水线ID
        
    Returns:
        流水线详情
    """
    try:
        user_id = get_jwt_identity()
        
        pipeline = preprocessing_service.get_pipeline(pipeline_id)
        
        return success_response(
            data=pipeline,
            message="获取流水线详情成功"
        )
    except DataPreprocessingError as e:
        return error_response(
            message=str(e),
            code=404,
            error_type="PIPELINE_NOT_FOUND"
        ), 404
    except Exception as e:
        logger.error(f"Error getting pipeline: {e}")
        return error_response(
            message=f"获取流水线详情时发生错误: {str(e)}",
            code=500,
            error_type="GET_PIPELINE_ERROR"
        ), 500


@data_preprocessing_bp.route('/pipelines/templates', methods=['GET'])
@jwt_required()
def get_public_templates():
    """获取公开的流水线模板
    
    Query Parameters:
        - limit: 返回数量限制
        
    Returns:
        公开模板列表
    """
    try:
        limit = min(int(request.args.get('limit', 50)), 100)
        
        from backend.repositories.data_preprocessing_repository import get_preprocessing_repository_manager
        repo_manager = get_preprocessing_repository_manager()
        
        templates = repo_manager.pipeline_repo.get_public_templates(limit=limit)
        
        return success_response(
            data=[t.to_dict() for t in templates],
            message="获取公开模板成功"
        )
    except Exception as e:
        logger.error(f"Error getting templates: {e}")
        return error_response(
            message=f"获取公开模板时发生错误: {str(e)}",
            code=500,
            error_type="GET_TEMPLATES_ERROR"
        ), 500


# ============================================================================
# 健康检查和统计API
# ============================================================================

@data_preprocessing_bp.route('/health', methods=['GET'])
def health_check():
    """健康检查
    
    Returns:
        服务健康状态
    """
    return success_response(
        data={
            "status": "healthy",
            "service": "data_preprocessing",
            "version": "1.0.0",
        },
        message="服务运行正常"
    )


@data_preprocessing_bp.route('/<dataset_id>/preprocessing/stats', methods=['GET'])
@jwt_required()
def get_preprocessing_stats(dataset_id: str):
    """获取数据集的预处理统计信息
    
    Args:
        dataset_id: 数据集ID
        
    Returns:
        统计信息
    """
    try:
        user_id = get_jwt_identity()
        
        # 获取任务统计
        tasks_result = preprocessing_service.list_tasks(
            dataset_id=dataset_id,
            page=1,
            page_size=1000  # 获取所有任务进行统计
        )
        
        tasks = tasks_result.get("tasks", [])
        
        stats = {
            "total_tasks": len(tasks),
            "completed_tasks": sum(1 for t in tasks if t.get("status") == "completed"),
            "failed_tasks": sum(1 for t in tasks if t.get("status") == "failed"),
            "pending_tasks": sum(1 for t in tasks if t.get("status") == "pending"),
            "task_type_counts": {},
            "latest_task": tasks[0] if tasks else None,
        }
        
        # 按任务类型统计
        for task in tasks:
            task_type = task.get("task_type", "unknown")
            stats["task_type_counts"][task_type] = stats["task_type_counts"].get(task_type, 0) + 1
        
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

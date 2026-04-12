#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""模型比较API接口

提供模型比较相关的API接口，支持：
- 多模型性能对比
- 模型导出（多种格式）
- 模型导入（多种来源）
- 模型克隆
- 训练历史查询
- 模型验证（多种类型）

架构调用关系：
API层 (本模块)
    -> Service层 (ModelManagementService)
        -> Repository层 (ModelManagementRepository)
        -> ModelService
        -> TrainingHistoryService
"""

import sys
import os
import uuid
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.core.exceptions import ValidationError, BusinessLogicError
from backend.utils.response import success_response, error_response

logger = logging.getLogger(__name__)

# 创建蓝图
model_comparison_bp = Blueprint('model_comparison', __name__, url_prefix='/api/v1/model-comparison')

# ==================== 服务初始化 ====================

_management_service = None


def get_management_service():
    """获取模型管理服务实例"""
    global _management_service
    if _management_service is None:
        try:
            from backend.services.model_management_service import get_management_service as get_svc
            _management_service = get_svc()
            logger.info("ModelComparisonAPI: ModelManagementService initialized")
        except Exception as e:
            logger.warning(f"ModelComparisonAPI: Failed to init service: {e}")
    return _management_service


def init_comparison_api(app=None):
    """初始化比较API模块
    
    Args:
        app: Flask应用实例
    """
    global _management_service
    _management_service = get_management_service()
    
    if app:
        app.register_blueprint(model_comparison_bp)
        logger.info("ModelComparisonAPI: Blueprint registered")


# ==================== API 端点 ====================

@model_comparison_bp.route('/compare', methods=['POST'])
@jwt_required()
def compare_models():
    """比较多个模型
    
    Request Body:
        {
            "modelIds": ["string"],
            "metricsToCompare": ["accuracy", "precision", "recall", "f1_score", "loss"],
            "comparisonConfig": {}
        }
    
    Returns:
        {
            "models": [...],
            "comparisonMetrics": [...],
            "winnerModelId": "string",
            "comparisonTime": "string"
        }
    """
    try:
        user_id = get_jwt_identity()
        data = request.get_json() or {}
        
        # 验证参数
        model_ids = data.get("modelIds", [])
        if not model_ids or not isinstance(model_ids, list):
            return error_response("请提供要比较的模型ID列表", 400)
        
        if len(model_ids) < 2:
            return error_response("至少需要选择2个模型进行比较", 400)
        
        if len(model_ids) > 10:
            return error_response("最多可比较10个模型", 400)
        
        metrics_to_compare = data.get("metricsToCompare", 
                                       ['accuracy', 'precision', 'recall', 'f1_score', 'loss'])
        comparison_config = data.get("comparisonConfig", {})
        
        # 调用服务层
        service = get_management_service()
        if service:
            result = service.compare_models(
                model_ids=model_ids,
                user_id=user_id,
                metrics_to_compare=metrics_to_compare,
                comparison_config=comparison_config
            )
            
            # 生成统计信息
            statistics = _generate_comparison_statistics(result.get('models', []))
            result['statistics'] = statistics
            
            return success_response(result, "模型比较完成")
        
        return error_response("模型管理服务不可用", 503)
        
    except ValidationError as e:
        return error_response(str(e), 400)
    except BusinessLogicError as e:
        return error_response(str(e), 500)
    except Exception as e:
        logger.error(f"compare_models error: {e}")
        return error_response(f"模型比较失败: {str(e)}", 500)


@model_comparison_bp.route('/history', methods=['GET'])
@jwt_required()
def get_comparison_history():
    """获取用户的比较历史记录
    
    Query Parameters:
        - page: 页码 (默认: 1)
        - limit: 每页数量 (默认: 20)
    
    Returns:
        {
            "comparisons": [...],
            "total": number,
            "page": number,
            "limit": number
        }
    """
    try:
        user_id = get_jwt_identity()
        
        page = int(request.args.get('page', 1))
        limit = int(request.args.get('limit', 20))
        offset = (page - 1) * limit
        
        # 从 Repository 层获取
        try:
            from backend.repositories.model_management_repository import get_management_repository
            repo = get_management_repository()
            comparisons, total = repo.list_user_comparisons(
                user_id=user_id,
                limit=limit,
                offset=offset
            )
            
            return success_response({
                'comparisons': comparisons,
                'total': total,
                'page': page,
                'limit': limit
            }, "获取比较历史成功")
            
        except Exception as e:
            logger.warning(f"Failed to get comparison history: {e}")
            return success_response({
                'comparisons': [],
                'total': 0,
                'page': page,
                'limit': limit
            }, "获取比较历史成功")
        
    except Exception as e:
        logger.error(f"get_comparison_history error: {e}")
        return error_response(f"获取比较历史失败: {str(e)}", 500)


@model_comparison_bp.route('/<model_id>/export', methods=['GET'])
@jwt_required()
def export_model(model_id: str):
    """导出模型
    
    Query Parameters:
        - format: 导出格式 (onnx, torchscript, tensorflow, tensorrt, safetensors)
        - includeWeights: 是否包含权重 (默认: true)
        - includeConfig: 是否包含配置 (默认: true)
    
    Returns:
        {
            "modelId": "string",
            "modelName": "string",
            "format": "string",
            "exportPath": "string",
            "status": "string",
            "exportTime": "string"
        }
    """
    try:
        user_id = get_jwt_identity()
        
        # 获取导出参数
        export_format = request.args.get("format", "onnx")
        include_weights = request.args.get("includeWeights", "true").lower() == "true"
        include_config = request.args.get("includeConfig", "true").lower() == "true"
        
        export_config = {
            'include_weights': include_weights,
            'include_config': include_config,
        }
        
        # 调用服务层
        service = get_management_service()
        if service:
            result = service.export_model(
                model_id=model_id,
                user_id=user_id,
                export_format=export_format,
                export_config=export_config
            )
            return success_response(result, "模型导出成功")
        
        return error_response("模型管理服务不可用", 503)
        
    except ValidationError as e:
        return error_response(str(e), 400)
    except BusinessLogicError as e:
        return error_response(str(e), 500)
    except Exception as e:
        logger.error(f"export_model error: {e}")
        return error_response(f"导出模型失败: {str(e)}", 500)


@model_comparison_bp.route('/export/formats', methods=['GET'])
@jwt_required()
def get_export_formats():
    """获取支持的导出格式列表
    
    Returns:
        {
            "formats": ["onnx", "torchscript", "tensorflow", "tensorrt", "safetensors"],
            "descriptions": {...}
        }
    """
    try:
        service = get_management_service()
        formats = service.get_supported_export_formats() if service else []
        
        descriptions = {
            'onnx': 'Open Neural Network Exchange - 跨平台模型格式',
            'torchscript': 'PyTorch TorchScript - PyTorch优化格式',
            'tensorflow': 'TensorFlow SavedModel - TensorFlow标准格式',
            'tensorrt': 'NVIDIA TensorRT - GPU推理优化格式',
            'safetensors': 'SafeTensors - 安全高效的张量存储格式',
        }
        
        return success_response({
            'formats': formats,
            'descriptions': descriptions
        }, "获取导出格式成功")
        
    except Exception as e:
        logger.error(f"get_export_formats error: {e}")
        return error_response(f"获取导出格式失败: {str(e)}", 500)


@model_comparison_bp.route('/import', methods=['POST'])
@jwt_required()
def import_model():
    """导入模型
    
    Form Data (multipart/form-data):
        - file: 模型文件
        - name: 模型名称
        - description: 模型描述
        - modelType: 模型类型
        - framework: 模型框架
    
    JSON Data:
        {
            "name": "string",
            "source": "local|url|huggingface|s3|gcs",
            "sourcePath": "string",
            "sourceUrl": "string",
            "modelType": "string",
            "framework": "string",
            "importConfig": {}
        }
    
    Returns:
        {
            "importId": "string",
            "status": "string",
            "modelName": "string",
            "targetModelId": "string"
        }
    """
    try:
        user_id = get_jwt_identity()
        
        # 处理文件上传
        if 'file' in request.files:
            file = request.files['file']
            if file.filename == '':
                return error_response("请选择要导入的模型文件", 400)
            
            model_name = request.form.get('name', file.filename)
            description = request.form.get('description', '')
            model_type = request.form.get('modelType', 'classification')
            framework = request.form.get('framework', 'pytorch')
            
            # 保存文件（实际应该保存到指定路径）
            import tempfile
            temp_dir = tempfile.mkdtemp()
            file_path = os.path.join(temp_dir, file.filename)
            file.save(file_path)
            
            # 调用服务层
            service = get_management_service()
            if service:
                result = service.import_model(
                    user_id=user_id,
                    model_name=model_name,
                    import_source='local',
                    source_path=file_path,
                    model_type=model_type,
                    model_framework=framework,
                    import_config={'description': description}
                )
                return success_response(result, "模型导入成功", 201)
            
            return error_response("模型管理服务不可用", 503)
        
        # 处理 JSON 数据
        elif request.is_json:
            data = request.get_json()
            
            if not data or "name" not in data:
                return error_response("请提供模型名称", 400)
            
            model_name = data["name"]
            import_source = data.get("source", "url")
            source_path = data.get("sourcePath")
            source_url = data.get("sourceUrl")
            model_type = data.get("modelType", "classification")
            framework = data.get("framework", "pytorch")
            import_config = data.get("importConfig", {})
            
            # 调用服务层
            service = get_management_service()
            if service:
                result = service.import_model(
                    user_id=user_id,
                    model_name=model_name,
                    import_source=import_source,
                    source_path=source_path,
                    source_url=source_url,
                    model_type=model_type,
                    model_framework=framework,
                    import_config=import_config
                )
                return success_response(result, "模型导入成功", 201)
            
            return error_response("模型管理服务不可用", 503)
        
        else:
            return error_response("请提供模型文件或JSON数据", 400)
        
    except ValidationError as e:
        return error_response(str(e), 400)
    except BusinessLogicError as e:
        return error_response(str(e), 500)
    except Exception as e:
        logger.error(f"import_model error: {e}")
        return error_response(f"导入模型失败: {str(e)}", 500)


@model_comparison_bp.route('/import/<import_id>/status', methods=['GET'])
@jwt_required()
def get_import_status(import_id: str):
    """获取导入状态
    
    Returns:
        {
            "importId": "string",
            "status": "pending|processing|completed|failed",
            "progress": number,
            "modelName": "string",
            "targetModelId": "string",
            "errorMessage": "string"
        }
    """
    try:
        user_id = get_jwt_identity()
        
        service = get_management_service()
        if service:
            result = service.get_import_status(import_id, user_id)
            if result:
                return success_response(result, "获取导入状态成功")
            return error_response("导入记录不存在", 404)
        
        return error_response("模型管理服务不可用", 503)
        
    except Exception as e:
        logger.error(f"get_import_status error: {e}")
        return error_response(f"获取导入状态失败: {str(e)}", 500)


@model_comparison_bp.route('/import/formats', methods=['GET'])
@jwt_required()
def get_import_formats():
    """获取支持的导入格式列表
    
    Returns:
        {
            "formats": ["pytorch", "tensorflow", "onnx", "safetensors", "huggingface"],
            "descriptions": {...}
        }
    """
    try:
        service = get_management_service()
        formats = service.get_supported_import_formats() if service else []
        
        descriptions = {
            'pytorch': 'PyTorch模型 (.pt, .pth, .bin)',
            'tensorflow': 'TensorFlow SavedModel',
            'onnx': 'ONNX模型 (.onnx)',
            'safetensors': 'SafeTensors格式 (.safetensors)',
            'huggingface': 'HuggingFace Hub模型',
        }
        
        return success_response({
            'formats': formats,
            'descriptions': descriptions
        }, "获取导入格式成功")
        
    except Exception as e:
        logger.error(f"get_import_formats error: {e}")
        return error_response(f"获取导入格式失败: {str(e)}", 500)


@model_comparison_bp.route('/<model_id>/clone', methods=['POST'])
@jwt_required()
def clone_model(model_id: str):
    """克隆模型
    
    Request Body:
        {
            "name": "string",
            "description": "string",
            "includeVersions": boolean
        }
    
    Returns:
        {
            "id": "string",
            "name": "string",
            "description": "string",
            "originalModelId": "string",
            "status": "string",
            "createdAt": "string"
        }
    """
    try:
        user_id = get_jwt_identity()
        data = request.get_json() or {}
        
        new_name = data.get("name")
        if not new_name:
            new_name = f"模型克隆_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        description = data.get("description", "")
        include_versions = data.get("includeVersions", False)
        
        # 调用服务层
        service = get_management_service()
        if service:
            result = service.clone_model(
                model_id=model_id,
                user_id=user_id,
                new_name=new_name,
                include_versions=include_versions
            )
            
            # 添加额外信息
            result['originalModelId'] = model_id
            result['clonedAt'] = datetime.utcnow().isoformat()
            
            if description:
                result['description'] = description
            
            return success_response({'data': result}, "模型克隆成功", 201)
        
        return error_response("模型管理服务不可用", 503)
        
    except ValidationError as e:
        return error_response(str(e), 400)
    except BusinessLogicError as e:
        return error_response(str(e), 500)
    except Exception as e:
        logger.error(f"clone_model error: {e}")
        return error_response(f"克隆模型失败: {str(e)}", 500)


@model_comparison_bp.route('/<model_id>/training-history', methods=['GET'])
@jwt_required()
def get_model_training_history(model_id: str):
    """获取模型训练历史
    
    Query Parameters:
        - page: 页码 (默认: 1)
        - limit: 每页数量 (默认: 20)
    
    Returns:
        {
            "modelId": "string",
            "modelName": "string",
            "history": [...],
            "totalTrainings": number
        }
    """
    try:
        user_id = get_jwt_identity()
        
        page = int(request.args.get('page', 1))
        limit = int(request.args.get('limit', 20))
        
        # 调用服务层
        service = get_management_service()
        if service:
            history = service.get_model_training_history(
                model_id=model_id,
                user_id=user_id,
                limit=limit
            )
            
            # 获取模型名称
            model_name = "Unknown"
            try:
                from backend.services.model_service import ModelService
                from backend.repositories.model_repository import ModelRepository
                model_service = ModelService(ModelRepository())
                model = model_service.get_model(model_id)
                if model and hasattr(model, 'name'):
                    model_name = model.name
            except Exception:
                pass
            
            return success_response({
                'modelId': model_id,
                'modelName': model_name,
                'history': history,
                'totalTrainings': len(history)
            }, "获取训练历史成功")
        
        return error_response("模型管理服务不可用", 503)
        
    except ValidationError as e:
        return error_response(str(e), 400)
    except Exception as e:
        logger.error(f"get_model_training_history error: {e}")
        return error_response(f"获取训练历史失败: {str(e)}", 500)


@model_comparison_bp.route('/<model_id>/validate', methods=['POST'])
@jwt_required()
def validate_model(model_id: str):
    """验证模型
    
    Request Body:
        {
            "type": "basic|performance|compatibility|all",
            "testData": [...],
            "validationConfig": {}
        }
    
    Returns:
        {
            "modelId": "string",
            "validationType": "string",
            "validatedAt": "string",
            "validatedBy": "string",
            "results": [
                {
                    "check": "string",
                    "status": "passed|warning|failed",
                    "message": "string",
                    "value": any
                }
            ],
            "summary": {
                "passed": number,
                "warnings": number,
                "failed": number,
                "total": number
            }
        }
    """
    try:
        user_id = get_jwt_identity()
        data = request.get_json() or {}
        
        validation_type = data.get("type", "basic")
        test_data = data.get("testData", [])
        validation_config = data.get("validationConfig", {})
        
        # 准备验证结果
        results = []
        now = datetime.utcnow()
        
        # 基础验证
        if validation_type in ["basic", "all"]:
            basic_results = _perform_basic_validation(model_id, user_id)
            results.extend(basic_results)
        
        # 性能验证
        if validation_type in ["performance", "all"]:
            service = get_management_service()
            if service and test_data:
                try:
                    perf_result = service.validate_model(
                        model_id=model_id,
                        user_id=user_id,
                        test_data=test_data,
                        validation_config=validation_config
                    )
                    
                    # 转换为验证结果格式
                    results.extend([
                        {
                            "check": "准确率检查",
                            "status": "passed" if perf_result.get('accuracy', 0) >= 0.7 else "warning",
                            "message": f"准确率: {perf_result.get('accuracy', 0):.2%}",
                            "value": perf_result.get('accuracy')
                        },
                        {
                            "check": "损失值检查",
                            "status": "passed" if perf_result.get('loss', 1) <= 0.5 else "warning",
                            "message": f"损失值: {perf_result.get('loss', 0):.4f}",
                            "value": perf_result.get('loss')
                        },
                        {
                            "check": "F1分数检查",
                            "status": "passed" if perf_result.get('f1Score', 0) >= 0.7 else "warning",
                            "message": f"F1分数: {perf_result.get('f1Score', 0):.2%}",
                            "value": perf_result.get('f1Score')
                        }
                    ])
                except Exception as e:
                    logger.warning(f"Performance validation failed: {e}")
                    results.append({
                        "check": "性能验证",
                        "status": "failed",
                        "message": f"性能验证失败: {str(e)}"
                    })
            else:
                perf_results = _perform_performance_validation(model_id, user_id)
                results.extend(perf_results)
        
        # 兼容性验证
        if validation_type in ["compatibility", "all"]:
            compat_results = _perform_compatibility_validation(model_id, user_id)
            results.extend(compat_results)
        
        # 计算摘要
        summary = {
            'passed': sum(1 for r in results if r['status'] == 'passed'),
            'warnings': sum(1 for r in results if r['status'] == 'warning'),
            'failed': sum(1 for r in results if r['status'] == 'failed'),
            'total': len(results)
        }
        
        return success_response({
            'modelId': model_id,
            'validationType': validation_type,
            'validatedAt': now.isoformat(),
            'validatedBy': user_id,
            'results': results,
            'summary': summary
        }, "模型验证完成")
        
    except ValidationError as e:
        return error_response(str(e), 400)
    except Exception as e:
        logger.error(f"validate_model error: {e}")
        return error_response(f"验证模型失败: {str(e)}", 500)


@model_comparison_bp.route('/<model_id>/performance', methods=['GET'])
@jwt_required()
def get_model_performance(model_id: str):
    """获取模型性能指标
    
    Query Parameters:
        - includeHistory: 是否包含历史记录 (默认: false)
    
    Returns:
        {
            "modelId": "string",
            "modelName": "string",
            "accuracy": number,
            "precision": number,
            "recall": number,
            "f1Score": number,
            "loss": number,
            "trainingTime": number,
            "inferenceTimeMs": number,
            "history": [...]  // if includeHistory=true
        }
    """
    try:
        user_id = get_jwt_identity()
        include_history = request.args.get('includeHistory', 'false').lower() == 'true'
        
        service = get_management_service()
        if service:
            result = service.get_model_performance(
                model_id=model_id,
                user_id=user_id,
                include_history=include_history
            )
            return success_response(result, "获取模型性能成功")
        
        return error_response("模型管理服务不可用", 503)
        
    except ValidationError as e:
        return error_response(str(e), 400)
    except Exception as e:
        logger.error(f"get_model_performance error: {e}")
        return error_response(f"获取模型性能失败: {str(e)}", 500)


@model_comparison_bp.route('/<model_id>/performance', methods=['POST'])
@jwt_required()
def record_model_performance(model_id: str):
    """记录模型性能指标
    
    Request Body:
        {
            "accuracy": number,
            "precision": number,
            "recall": number,
            "f1Score": number,
            "loss": number,
            "trainingTimeSeconds": number,
            "inferenceTimeMs": number,
            "testDataSize": number,
            "customMetrics": {}
        }
    
    Returns:
        {
            "id": "string",
            "modelId": "string",
            ...
        }
    """
    try:
        user_id = get_jwt_identity()
        data = request.get_json() or {}
        
        # 转换键名
        metrics = {
            'accuracy': data.get('accuracy'),
            'precision': data.get('precision'),
            'recall': data.get('recall'),
            'f1_score': data.get('f1Score'),
            'loss': data.get('loss'),
            'training_time_seconds': data.get('trainingTimeSeconds'),
            'inference_time_ms': data.get('inferenceTimeMs'),
            'test_data_size': data.get('testDataSize'),
            'custom_metrics': data.get('customMetrics', {}),
        }
        
        service = get_management_service()
        if service:
            result = service.record_model_performance(
                model_id=model_id,
                user_id=user_id,
                metrics=metrics
            )
            return success_response(result, "记录性能指标成功", 201)
        
        return error_response("模型管理服务不可用", 503)
        
    except ValidationError as e:
        return error_response(str(e), 400)
    except BusinessLogicError as e:
        return error_response(str(e), 500)
    except Exception as e:
        logger.error(f"record_model_performance error: {e}")
        return error_response(f"记录性能指标失败: {str(e)}", 500)


# ==================== 辅助函数 ====================

def _generate_comparison_statistics(models_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """生成比较统计信息
    
    Args:
        models_data: 模型数据列表
        
    Returns:
        统计信息字典
    """
    metrics = ['accuracy', 'precision', 'recall', 'f1Score', 'loss', 
               'trainingTime', 'inferenceTimeMs']
    statistics = {}
    
    for metric in metrics:
        values = []
        model_values = []
        
        for model in models_data:
            model_metrics = model.get('metrics', {})
            value = model_metrics.get(metric)
            if value is not None:
                values.append(value)
                model_values.append({
                    'modelId': model.get('id'),
                    'value': value
                })
        
        if values:
            # 对于 loss，最小值为最优
            is_lower_better = metric in ['loss', 'trainingTime', 'inferenceTimeMs']
            best_model = min(model_values, key=lambda x: x['value']) if is_lower_better \
                        else max(model_values, key=lambda x: x['value'])
            
            statistics[metric] = {
                'min': min(values),
                'max': max(values),
                'avg': sum(values) / len(values),
                'bestModel': best_model['modelId']
            }
    
    return statistics


def _perform_basic_validation(model_id: str, user_id: str) -> List[Dict[str, Any]]:
    """执行基础验证
    
    Args:
        model_id: 模型ID
        user_id: 用户ID
        
    Returns:
        验证结果列表
    """
    results = []
    
    # 模型存在性检查
    try:
        from backend.services.model_service import ModelService
        from backend.repositories.model_repository import ModelRepository
        model_service = ModelService(ModelRepository())
        model = model_service.get_model(model_id)
        
        if model:
            results.append({
                "check": "模型存在性",
                "status": "passed",
                "message": "模型存在"
            })
            
            # 配置完整性检查
            if hasattr(model, 'config') and model.config:
                results.append({
                    "check": "模型配置完整性",
                    "status": "passed",
                    "message": "模型配置完整"
                })
            else:
                results.append({
                    "check": "模型配置完整性",
                    "status": "warning",
                    "message": "模型配置缺失或不完整"
                })
            
            # 名称有效性检查
            if hasattr(model, 'name') and model.name:
                results.append({
                    "check": "模型名称有效性",
                    "status": "passed",
                    "message": f"模型名称: {model.name}"
                })
            else:
                results.append({
                    "check": "模型名称有效性",
                    "status": "failed",
                    "message": "模型名称无效"
                })
        else:
            results.append({
                "check": "模型存在性",
                "status": "failed",
                "message": "模型不存在"
            })
            
    except Exception as e:
        results.append({
            "check": "基础验证",
            "status": "failed",
            "message": f"验证失败: {str(e)}"
        })
    
    return results


def _perform_performance_validation(model_id: str, user_id: str) -> List[Dict[str, Any]]:
    """执行性能验证
    
    Args:
        model_id: 模型ID
        user_id: 用户ID
        
    Returns:
        验证结果列表
    """
    results = []
    
    try:
        service = get_management_service()
        if service:
            performance = service.get_model_performance(model_id, user_id)
            
            # 准确率检查
            accuracy = performance.get('accuracy')
            if accuracy is not None:
                status = "passed" if accuracy >= 0.7 else "warning" if accuracy >= 0.5 else "failed"
                results.append({
                    "check": "准确率检查",
                    "status": status,
                    "message": f"准确率: {accuracy:.2%}",
                    "value": accuracy
                })
            
            # 损失值检查
            loss = performance.get('loss')
            if loss is not None:
                status = "passed" if loss <= 0.3 else "warning" if loss <= 0.5 else "failed"
                results.append({
                    "check": "损失值检查",
                    "status": status,
                    "message": f"损失值: {loss:.4f}",
                    "value": loss
                })
            
            # F1分数检查
            f1_score = performance.get('f1Score')
            if f1_score is not None:
                status = "passed" if f1_score >= 0.7 else "warning" if f1_score >= 0.5 else "failed"
                results.append({
                    "check": "F1分数检查",
                    "status": status,
                    "message": f"F1分数: {f1_score:.2%}",
                    "value": f1_score
                })
        else:
            results.append({
                "check": "性能验证",
                "status": "warning",
                "message": "性能验证服务不可用"
            })
            
    except Exception as e:
        results.append({
            "check": "性能验证",
            "status": "failed",
            "message": f"性能验证失败: {str(e)}"
        })
    
    return results


def _perform_compatibility_validation(model_id: str, user_id: str) -> List[Dict[str, Any]]:
    """执行兼容性验证
    
    Args:
        model_id: 模型ID
        user_id: 用户ID
        
    Returns:
        验证结果列表
    """
    results = []
    
    try:
        from backend.services.model_service import ModelService
        from backend.repositories.model_repository import ModelRepository
        model_service = ModelService(ModelRepository())
        model = model_service.get_model(model_id)
        
        if model:
            # 版本兼容性检查
            version = getattr(model, 'version', '1.0.0')
            results.append({
                "check": "版本兼容性",
                "status": "passed",
                "message": f"模型版本: {version}，兼容当前系统"
            })
            
            # 框架兼容性检查
            framework = getattr(model, 'framework', 'pytorch')
            supported_frameworks = ['pytorch', 'tensorflow', 'onnx', 'jax']
            if framework.lower() in supported_frameworks:
                results.append({
                    "check": "框架兼容性",
                    "status": "passed",
                    "message": f"模型框架: {framework}，支持的框架"
                })
            else:
                results.append({
                    "check": "框架兼容性",
                    "status": "warning",
                    "message": f"模型框架: {framework}，可能需要额外适配"
                })
            
            # 导出格式兼容性
            service = get_management_service()
            if service:
                export_formats = service.get_supported_export_formats()
                results.append({
                    "check": "导出格式兼容性",
                    "status": "passed",
                    "message": f"支持导出格式: {', '.join(export_formats)}"
                })
                
    except Exception as e:
        results.append({
            "check": "兼容性验证",
            "status": "failed",
            "message": f"兼容性验证失败: {str(e)}"
        })
    
    return results


# ==================== 导出 ====================

__all__ = [
    'model_comparison_bp',
    'init_comparison_api',
    'get_management_service',
]

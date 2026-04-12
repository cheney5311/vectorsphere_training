"""
模型评估API
提供模型评估和对比的REST API接口，支持租户隔离和持久化
"""
from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
from typing import Dict, Any
import logging

from backend.services.model_evaluation_service import (
    ModelEvaluationService, 
    EvaluationMetricType,
    EvaluationResult,
    ModelComparison
)

# 创建蓝图
model_evaluation_bp = Blueprint('model_evaluation', __name__, url_prefix='/api/v1/training/model')
logger = logging.getLogger(__name__)

# 初始化服务（默认使用内存存储，生产环境可改为数据库）
evaluation_service = ModelEvaluationService(use_memory_storage=True)


@model_evaluation_bp.route('/evaluate', methods=['POST'])
@jwt_required()
def evaluate_model():
    """
    评估模型性能
    
    Request Body:
        model_id: 模型ID（必需）
        dataset_id: 数据集ID（必需）
        evaluation_config: 评估配置（可选）
            - validation_strategy: 验证策略 (holdout, cross_validation)
            - metrics: 指标列表 (accuracy, precision, recall, f1_score, auc)
            - cross_validation_folds: 交叉验证折数
            - test_size: 测试集比例
        
    Returns:
        评估结果，包括各项指标和评估ID
    """
    try:
        # 获取当前用户ID
        user_id = get_jwt_identity()
        
        # 获取租户ID
        tenant_id = request.headers.get('X-Tenant-ID') or request.args.get('tenant_id')
        
        # 获取请求数据
        data = request.get_json()
        model_id = data.get('model_id')
        dataset_id = data.get('dataset_id')
        evaluation_config = data.get('evaluation_config', {})
        
        logger.info(f"用户 {user_id} 请求评估模型 {model_id}, tenant_id={tenant_id}")
        
        if not model_id:
            return jsonify({
                'success': False,
                'error': '缺少模型ID'
            }), 400
        
        if not dataset_id:
            return jsonify({
                'success': False,
                'error': '缺少数据集ID'
            }), 400
        
        # 执行模型评估
        result: EvaluationResult = evaluation_service.automated_evaluation(
            model_id=model_id, 
            dataset_id=dataset_id, 
            evaluation_config=evaluation_config,
            tenant_id=tenant_id,
            user_id=user_id
        )
        
        # 返回结果
        return jsonify({
            'success': True,
            'data': {
                'evaluation_id': result.metadata.get('evaluation_id') if result.metadata else None,
                'model_id': result.model_id,
                'dataset_id': result.dataset_id,
                'metrics': [
                    {
                        'name': metric.name,
                        'value': metric.value,
                        'type': metric.type.value,
                        'description': metric.description
                    } for metric in result.metrics
                ],
                'evaluation_config': result.evaluation_config,
                'timestamp': result.timestamp,
                'duration_seconds': result.metadata.get('duration_seconds') if result.metadata else None
            }
        }), 200
        
    except Exception as e:
        logger.error(f"模型评估失败: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'模型评估失败: {str(e)}'
        }), 500


@model_evaluation_bp.route('/evaluation/results', methods=['GET'])
@jwt_required()
def get_evaluation_results():
    """
    获取评估结果列表
    
    Query Parameters:
        model_id: 模型ID（可选）
        evaluation_id: 评估ID（可选）
        status: 状态过滤 (pending, running, completed, failed)
        limit: 返回数量限制，默认100
        offset: 偏移量，默认0
        
    Returns:
        评估结果列表
    """
    try:
        # 获取当前用户ID
        user_id = get_jwt_identity()
        
        # 获取租户ID
        tenant_id = request.headers.get('X-Tenant-ID') or request.args.get('tenant_id')
        
        # 获取查询参数
        model_id = request.args.get('model_id')
        evaluation_id = request.args.get('evaluation_id')
        status = request.args.get('status')
        limit = request.args.get('limit', 100, type=int)
        offset = request.args.get('offset', 0, type=int)
        
        logger.info(
            f"用户 {user_id} 请求获取评估结果，"
            f"model_id={model_id}, evaluation_id={evaluation_id}"
        )
        
        # 调用服务层获取评估结果
        results = evaluation_service.get_evaluation_results(
            model_id=model_id,
            evaluation_id=evaluation_id,
            tenant_id=tenant_id,
            user_id=user_id,
            status=status,
            limit=limit,
            offset=offset
        )
        
        return jsonify({
            'success': True,
            'data': results
        }), 200
        
    except Exception as e:
        logger.error(f"获取评估结果失败: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'获取评估结果失败: {str(e)}'
        }), 500


@model_evaluation_bp.route('/evaluation/<evaluation_id>', methods=['GET'])
@jwt_required()
def get_evaluation_detail(evaluation_id: str):
    """
    获取单个评估结果详情
    
    Args:
        evaluation_id: 评估ID
        
    Query Parameters:
        include_metrics: 是否包含详细指标，默认true
        
    Returns:
        评估详情
    """
    try:
        user_id = get_jwt_identity()
        include_metrics = request.args.get('include_metrics', 'true').lower() == 'true'
        
        logger.info(f"用户 {user_id} 请求获取评估详情: {evaluation_id}")
        
        evaluation = evaluation_service.get_evaluation_by_id(
            evaluation_id=evaluation_id,
            include_metrics=include_metrics
        )
        
        if not evaluation:
            return jsonify({
                'success': False,
                'error': f'评估记录不存在: {evaluation_id}'
            }), 404
        
        return jsonify({
            'success': True,
            'data': evaluation
        }), 200
        
    except Exception as e:
        logger.error(f"获取评估详情失败: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'获取评估详情失败: {str(e)}'
        }), 500


@model_evaluation_bp.route('/evaluation/<evaluation_id>', methods=['DELETE'])
@jwt_required()
def delete_evaluation(evaluation_id: str):
    """
    删除评估记录
    
    Args:
        evaluation_id: 评估ID
        
    Returns:
        删除结果
    """
    try:
        user_id = get_jwt_identity()
        tenant_id = request.headers.get('X-Tenant-ID') or request.args.get('tenant_id')
        
        logger.info(f"用户 {user_id} 请求删除评估: {evaluation_id}")
        
        success = evaluation_service.delete_evaluation(
            evaluation_id=evaluation_id,
            tenant_id=tenant_id
        )
        
        if success:
            return jsonify({
                'success': True,
                'message': f'评估记录已删除: {evaluation_id}'
            }), 200
        else:
            return jsonify({
                'success': False,
                'error': f'删除评估记录失败: {evaluation_id}'
            }), 404
            
    except Exception as e:
        logger.error(f"删除评估记录失败: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'删除评估记录失败: {str(e)}'
        }), 500


@model_evaluation_bp.route('/models/compare', methods=['POST'])
@jwt_required()
def compare_models():
    """
    对比多个模型
    
    Request Body:
        model_ids: 模型ID列表（必需，至少2个）
        dataset_id: 数据集ID（必需）
        comparison_config: 对比配置（可选）
            - comparison_metrics: 对比指标列表
            - decision_criteria: 决策标准 (multi_objective, single_metric, weighted)
            - business_constraints: 业务约束
        
    Returns:
        对比结果，包括获胜模型、推荐和风险评估
    """
    try:
        # 获取当前用户ID
        user_id = get_jwt_identity()
        tenant_id = request.headers.get('X-Tenant-ID') or request.args.get('tenant_id')
        
        logger.info(f"用户 {user_id} 请求对比模型, tenant_id={tenant_id}")
        
        # 获取请求数据
        data = request.get_json()
        model_ids = data.get('model_ids', [])
        dataset_id = data.get('dataset_id')
        comparison_config = data.get('comparison_config', {})
        
        if not model_ids or not dataset_id:
            return jsonify({
                'success': False,
                'error': '缺少模型ID列表或数据集ID'
            }), 400
        
        if len(model_ids) < 2:
            return jsonify({
                'success': False,
                'error': '至少需要两个模型进行对比'
            }), 400
        
        # 执行模型对比
        comparison: ModelComparison = evaluation_service.model_comparison(
            model_ids=model_ids, 
            dataset_id=dataset_id, 
            comparison_config=comparison_config,
            tenant_id=tenant_id,
            user_id=user_id
        )
        
        # 返回结果
        return jsonify({
            'success': True,
            'data': {
                'winner_model_id': comparison.winner_model_id,
                'recommendations': comparison.recommendations,
                'risk_assessment': comparison.risk_assessment,
                'comparison_metrics': [
                    {
                        'name': metric.name,
                        'value': metric.value,
                        'type': metric.type.value,
                        'description': metric.description
                    } for metric in comparison.comparison_metrics
                ]
            }
        }), 200
        
    except Exception as e:
        logger.error(f"模型对比失败: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'模型对比失败: {str(e)}'
        }), 500


@model_evaluation_bp.route('/comparison/<comparison_id>', methods=['GET'])
@jwt_required()
def get_comparison_detail(comparison_id: str):
    """
    获取对比记录详情
    
    Args:
        comparison_id: 对比ID
        
    Returns:
        对比详情
    """
    try:
        user_id = get_jwt_identity()
        
        logger.info(f"用户 {user_id} 请求获取对比详情: {comparison_id}")
        
        comparison = evaluation_service.get_comparison_by_id(comparison_id)
        
        if not comparison:
            return jsonify({
                'success': False,
                'error': f'对比记录不存在: {comparison_id}'
            }), 404
        
        return jsonify({
            'success': True,
            'data': comparison
        }), 200
        
    except Exception as e:
        logger.error(f"获取对比详情失败: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'获取对比详情失败: {str(e)}'
        }), 500


@model_evaluation_bp.route('/comparison/history', methods=['GET'])
@jwt_required()
def get_comparison_history():
    """
    获取对比历史
    
    Query Parameters:
        status: 状态过滤
        limit: 返回数量限制，默认20
        
    Returns:
        对比历史列表
    """
    try:
        user_id = get_jwt_identity()
        tenant_id = request.headers.get('X-Tenant-ID') or request.args.get('tenant_id')
        
        status = request.args.get('status')
        limit = request.args.get('limit', 20, type=int)
        
        logger.info(f"用户 {user_id} 请求获取对比历史")
        
        history = evaluation_service.get_comparison_history(
            user_id=user_id,
            tenant_id=tenant_id,
            status=status,
            limit=limit
        )
        
        return jsonify({
            'success': True,
            'data': {
                'comparisons': history,
                'total': len(history)
            }
        }), 200
        
    except Exception as e:
        logger.error(f"获取对比历史失败: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'获取对比历史失败: {str(e)}'
        }), 500


@model_evaluation_bp.route('/comparison/<comparison_id>', methods=['DELETE'])
@jwt_required()
def delete_comparison(comparison_id: str):
    """
    删除对比记录
    
    Args:
        comparison_id: 对比ID
        
    Returns:
        删除结果
    """
    try:
        user_id = get_jwt_identity()
        tenant_id = request.headers.get('X-Tenant-ID') or request.args.get('tenant_id')
        
        logger.info(f"用户 {user_id} 请求删除对比: {comparison_id}")
        
        success = evaluation_service.delete_comparison(
            comparison_id=comparison_id,
            tenant_id=tenant_id
        )
        
        if success:
            return jsonify({
                'success': True,
                'message': f'对比记录已删除: {comparison_id}'
            }), 200
        else:
            return jsonify({
                'success': False,
                'error': f'删除对比记录失败: {comparison_id}'
            }), 404
            
    except Exception as e:
        logger.error(f"删除对比记录失败: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'删除对比记录失败: {str(e)}'
        }), 500


@model_evaluation_bp.route('/metrics/types', methods=['GET'])
@jwt_required()
def get_metric_types():
    """
    获取支持的评估指标类型
    
    Returns:
        指标类型列表及其描述
    """
    try:
        metric_types = []
        for metric in EvaluationMetricType:
            metric_types.append({
                'type': metric.value,
                'name': metric.name,
                'description': _get_metric_description(metric.value)
            })
        
        return jsonify({
            'success': True,
            'data': {
                'metric_types': metric_types
            }
        }), 200
        
    except Exception as e:
        logger.error(f"获取指标类型失败: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'获取指标类型失败: {str(e)}'
        }), 500


def _get_metric_description(metric_type: str) -> str:
    """获取指标描述"""
    descriptions = {
        'accuracy': '准确率，正确预测的样本占总样本的比例',
        'precision': '精确率，预测为正的样本中实际为正的比例',
        'recall': '召回率，实际为正的样本中被正确预测为正的比例',
        'f1_score': 'F1分数，精确率和召回率的调和平均数',
        'auc': 'AUC值，ROC曲线下的面积',
        'roc': 'ROC曲线，真正率与假正率的关系曲线',
        'bleu': 'BLEU分数，机器翻译评估指标',
        'rouge': 'ROUGE分数，文本摘要评估指标',
        'custom': '自定义指标'
    }
    return descriptions.get(metric_type, '未知指标')


@model_evaluation_bp.route('/models/<model_id>/evaluation-history', methods=['GET'])
@jwt_required()
def get_evaluation_history(model_id: str):
    """
    获取模型评估历史
    
    Args:
        model_id: 模型ID
        
    Query Parameters:
        limit: 限制返回记录数，默认10
        dataset_id: 数据集ID过滤
        
    Returns:
        评估历史记录
    """
    try:
        # 获取当前用户ID
        user_id = get_jwt_identity()
        tenant_id = request.headers.get('X-Tenant-ID') or request.args.get('tenant_id')
        
        logger.info(f"用户 {user_id} 请求获取模型 {model_id} 的评估历史")
        
        # 获取查询参数
        limit = request.args.get('limit', 10, type=int)
        dataset_id = request.args.get('dataset_id')
        
        # 调用服务获取评估历史
        history = evaluation_service.get_evaluation_history(
            model_id=model_id,
            dataset_id=dataset_id,
            tenant_id=tenant_id,
            limit=limit
        )
        
        return jsonify({
            'success': True,
            'data': {
                'history': history,
                'total': len(history),
                'model_id': model_id
            }
        }), 200
        
    except Exception as e:
        logger.error(f"获取评估历史失败: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'获取评估历史失败: {str(e)}'
        }), 500


@model_evaluation_bp.route('/evaluation/statistics', methods=['GET'])
@jwt_required()
def get_evaluation_statistics():
    """
    获取评估统计信息
    
    Query Parameters:
        model_id: 模型ID（可选，不提供则统计租户下所有）
        
    Returns:
        统计信息
    """
    try:
        user_id = get_jwt_identity()
        tenant_id = request.headers.get('X-Tenant-ID') or request.args.get('tenant_id')
        
        if not tenant_id:
            return jsonify({
                'success': False,
                'error': '获取评估统计需要提供租户ID'
            }), 400
        
        model_id = request.args.get('model_id')
        
        logger.info(
            f"用户 {user_id} 请求获取评估统计, "
            f"tenant_id={tenant_id}, model_id={model_id}"
        )
        
        stats = evaluation_service.get_evaluation_statistics(
            tenant_id=tenant_id,
            model_id=model_id
        )
        
        return jsonify({
            'success': True,
            'data': stats
        }), 200
        
    except Exception as e:
        logger.error(f"获取评估统计失败: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'获取评估统计失败: {str(e)}'
        }), 500


@model_evaluation_bp.route('/evaluate/batch', methods=['POST'])
@jwt_required()
def batch_evaluate_models():
    """
    批量评估多个模型
    
    Request Body:
        model_ids: 模型ID列表（必需）
        dataset_id: 数据集ID（必需）
        evaluation_config: 评估配置（可选）
        
    Returns:
        批量评估结果
    """
    try:
        user_id = get_jwt_identity()
        tenant_id = request.headers.get('X-Tenant-ID') or request.args.get('tenant_id')
        
        data = request.get_json()
        model_ids = data.get('model_ids', [])
        dataset_id = data.get('dataset_id')
        evaluation_config = data.get('evaluation_config')
        
        if not model_ids:
            return jsonify({
                'success': False,
                'error': '缺少模型ID列表'
            }), 400
        
        if not dataset_id:
            return jsonify({
                'success': False,
                'error': '缺少数据集ID'
            }), 400
        
        logger.info(
            f"用户 {user_id} 请求批量评估 {len(model_ids)} 个模型"
        )
        
        results = evaluation_service.batch_evaluate(
            model_ids=model_ids,
            dataset_id=dataset_id,
            evaluation_config=evaluation_config,
            tenant_id=tenant_id,
            user_id=user_id
        )
        
        # 统计结果
        completed = sum(1 for r in results if r['status'] == 'completed')
        failed = sum(1 for r in results if r['status'] == 'failed')
        
        return jsonify({
            'success': True,
            'data': {
                'results': results,
                'summary': {
                    'total': len(results),
                    'completed': completed,
                    'failed': failed
                }
            }
        }), 200
        
    except Exception as e:
        logger.error(f"批量评估失败: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'批量评估失败: {str(e)}'
        }), 500


@model_evaluation_bp.route('/models/best', methods=['POST'])
@jwt_required()
def get_best_model():
    """
    获取最佳模型
    
    Request Body:
        model_ids: 模型ID列表（必需）
        dataset_id: 数据集ID（必需）
        metric_name: 评估指标名称（可选，默认accuracy）
        
    Returns:
        最佳模型信息
    """
    try:
        user_id = get_jwt_identity()
        tenant_id = request.headers.get('X-Tenant-ID') or request.args.get('tenant_id')
        
        data = request.get_json()
        model_ids = data.get('model_ids', [])
        dataset_id = data.get('dataset_id')
        metric_name = data.get('metric_name', 'accuracy')
        
        if not model_ids or not dataset_id:
            return jsonify({
                'success': False,
                'error': '缺少模型ID列表或数据集ID'
            }), 400
        
        logger.info(
            f"用户 {user_id} 请求获取最佳模型, metric={metric_name}"
        )
        
        best = evaluation_service.get_best_model(
            model_ids=model_ids,
            dataset_id=dataset_id,
            metric_name=metric_name,
            tenant_id=tenant_id
        )
        
        if best:
            return jsonify({
                'success': True,
                'data': best
            }), 200
        else:
            return jsonify({
                'success': False,
                'error': '未找到合适的模型'
            }), 404
            
    except Exception as e:
        logger.error(f"获取最佳模型失败: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'获取最佳模型失败: {str(e)}'
        }), 500

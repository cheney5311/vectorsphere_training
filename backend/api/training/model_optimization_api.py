"""
模型优化API
提供模型压缩和推理优化的REST API接口，支持租户隔离和持久化
"""
from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
from typing import Dict, Any
import logging

from backend.services.model_optimization_service import (
    ModelOptimizationService,
    OptimizationTechnique,
    CompressionStrategy,
    OptimizationConfig,
    InferenceOptimizationConfig
)

# 创建蓝图
model_optimization_bp = Blueprint('model_optimization', __name__, url_prefix='/api/training/optimization')
logger = logging.getLogger(__name__)

# 初始化服务（默认使用内存存储）
optimization_service = ModelOptimizationService(use_memory_storage=True)


@model_optimization_bp.route('/models/<model_id>/compress', methods=['POST'])
@jwt_required()
def compress_model(model_id: str):
    """
    模型压缩
    
    Args:
        model_id: 模型ID
        
    Request Body:
        technique: 压缩技术 (quantization, pruning, knowledge_distillation, low_rank_decomposition)
        compression_ratio: 压缩率 (0-1)
        strategy: 压缩策略 (structured, unstructured, mixed)
        quantization_bits: 量化位数
        preserve_accuracy: 是否保持精度
        
    Returns:
        压缩结果
    """
    try:
        # 获取当前用户ID
        user_id = get_jwt_identity()
        tenant_id = request.headers.get('X-Tenant-ID') or request.args.get('tenant_id')
        
        logger.info(f"用户 {user_id} 请求压缩模型 {model_id}, tenant_id={tenant_id}")
        
        # 获取请求数据
        data = request.get_json() or {}
        technique_str = data.get('technique', 'pruning')
        compression_ratio = data.get('compression_ratio', 0.5)
        strategy_str = data.get('strategy')
        quantization_bits = data.get('quantization_bits', 8)
        preserve_accuracy = data.get('preserve_accuracy', True)
        
        # 验证技术类型
        try:
            technique = OptimizationTechnique(technique_str)
        except ValueError:
            return jsonify({
                'success': False,
                'error': f'不支持的压缩技术: {technique_str}',
                'supported_techniques': [t.value for t in OptimizationTechnique]
            }), 400
        
        # 验证策略类型
        strategy = None
        if strategy_str:
            try:
                strategy = CompressionStrategy(strategy_str)
            except ValueError:
                return jsonify({
                    'success': False,
                    'error': f'不支持的压缩策略: {strategy_str}',
                    'supported_strategies': [s.value for s in CompressionStrategy]
                }), 400
        
        # 创建配置
        config = OptimizationConfig(
            technique=technique,
            compression_ratio=compression_ratio,
            strategy=strategy,
            quantization_bits=quantization_bits,
            preserve_accuracy=preserve_accuracy
        )
        
        # 执行模型压缩
        result = optimization_service.model_compression(
            model_id=model_id, 
            config=config,
            tenant_id=tenant_id,
            user_id=user_id
        )
        
        # 返回结果
        return jsonify({
            'success': True,
            'data': {
                'original_model_id': result.original_model_id,
                'optimized_model_id': result.optimized_model_id,
                'technique': result.technique,
                'compression_ratio': result.compression_ratio,
                'accuracy_preserved': result.accuracy_preserved,
                'model_size_reduction': result.model_size_reduction,
                'inference_speedup': result.inference_speedup,
                'optimization_time': result.optimization_time,
                'metrics': result.metrics
            }
        }), 200
        
    except Exception as e:
        logger.error(f"模型压缩失败: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'模型压缩失败: {str(e)}'
        }), 500


@model_optimization_bp.route('/models/<model_id>/optimize-inference', methods=['POST'])
@jwt_required()
def optimize_model_inference(model_id: str):
    """
    推理优化
    
    Args:
        model_id: 模型ID
        
    Request Body:
        graph_optimization: 图优化
        operator_fusion: 算子融合
        constant_folding: 常量折叠
        dead_code_elimination: 死代码消除
        memory_optimization: 内存优化
        hardware_target: 硬件目标 (cpu, gpu, tpu, edge)
        
    Returns:
        推理优化结果
    """
    try:
        # 获取当前用户ID
        user_id = get_jwt_identity()
        tenant_id = request.headers.get('X-Tenant-ID') or request.args.get('tenant_id')
        
        logger.info(f"用户 {user_id} 请求推理优化模型 {model_id}, tenant_id={tenant_id}")
        
        # 获取请求数据
        data = request.get_json() or {}
        config = InferenceOptimizationConfig(
            graph_optimization=data.get('graph_optimization', True),
            operator_fusion=data.get('operator_fusion', True),
            constant_folding=data.get('constant_folding', True),
            dead_code_elimination=data.get('dead_code_elimination', True),
            memory_optimization=data.get('memory_optimization', True),
            hardware_target=data.get('hardware_target', 'cpu')
        )
        
        # 执行推理优化
        result = optimization_service.inference_optimization(
            model_id=model_id, 
            config=config,
            tenant_id=tenant_id,
            user_id=user_id
        )
        
        # 返回结果
        return jsonify({
            'success': True,
            'data': {
                'original_model_id': result.original_model_id,
                'optimized_model_id': result.optimized_model_id,
                'optimization_config': {
                    'graph_optimization': result.optimization_config.graph_optimization,
                    'operator_fusion': result.optimization_config.operator_fusion,
                    'constant_folding': result.optimization_config.constant_folding,
                    'dead_code_elimination': result.optimization_config.dead_code_elimination,
                    'memory_optimization': result.optimization_config.memory_optimization,
                    'hardware_target': result.optimization_config.hardware_target
                },
                'latency_reduction': result.latency_reduction,
                'memory_usage_reduction': result.memory_usage_reduction,
                'throughput_improvement': result.throughput_improvement,
                'optimization_time': result.optimization_time,
                'metrics': result.metrics
            }
        }), 200
        
    except Exception as e:
        logger.error(f"推理优化失败: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'推理优化失败: {str(e)}'
        }), 500


@model_optimization_bp.route('/models/<model_id>/auto-optimize', methods=['POST'])
@jwt_required()
def auto_optimize_model(model_id: str):
    """
    自动优化
    
    Args:
        model_id: 模型ID
        
    Request Body:
        target_constraints: 目标约束条件
            - size_reduction: 目标大小减少率 (0-1)
            - accuracy_preservation: 精度保持要求 (0-1)
            - speed_improvement: 速度提升目标
        
    Returns:
        优化结果
    """
    try:
        # 获取当前用户ID
        user_id = get_jwt_identity()
        tenant_id = request.headers.get('X-Tenant-ID') or request.args.get('tenant_id')
        
        logger.info(f"用户 {user_id} 请求自动优化模型 {model_id}, tenant_id={tenant_id}")
        
        # 获取请求数据
        data = request.get_json() or {}
        target_constraints = data.get('target_constraints', {})
        
        # 执行自动优化
        result = optimization_service.auto_optimization(
            model_id=model_id, 
            target_constraints=target_constraints,
            tenant_id=tenant_id,
            user_id=user_id
        )
        
        # 返回结果
        return jsonify({
            'success': True,
            'data': {
                'original_model_id': result.original_model_id,
                'optimized_model_id': result.optimized_model_id,
                'technique': result.technique,
                'compression_ratio': result.compression_ratio,
                'accuracy_preserved': result.accuracy_preserved,
                'model_size_reduction': result.model_size_reduction,
                'inference_speedup': result.inference_speedup,
                'optimization_time': result.optimization_time,
                'metrics': result.metrics,
                'auto_selected': True
            }
        }), 200
        
    except Exception as e:
        logger.error(f"自动优化失败: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'自动优化失败: {str(e)}'
        }), 500


@model_optimization_bp.route('/techniques', methods=['GET'])
@jwt_required()
def get_optimization_techniques():
    """
    获取支持的优化技术
    
    Returns:
        优化技术列表及其描述
    """
    try:
        techniques = []
        for technique in OptimizationTechnique:
            techniques.append({
                'value': technique.value,
                'name': technique.name,
                'description': _get_technique_description(technique.value)
            })
        
        strategies = []
        for strategy in CompressionStrategy:
            strategies.append({
                'value': strategy.value,
                'name': strategy.name,
                'description': _get_strategy_description(strategy.value)
            })
        
        hardware_targets = [
            {'value': 'cpu', 'name': 'CPU', 'description': '通用CPU优化'},
            {'value': 'gpu', 'name': 'GPU', 'description': 'NVIDIA/AMD GPU优化'},
            {'value': 'tpu', 'name': 'TPU', 'description': 'Google TPU优化'},
            {'value': 'edge', 'name': 'Edge', 'description': '边缘设备优化'}
        ]
        
        return jsonify({
            'success': True,
            'data': {
                'techniques': techniques,
                'strategies': strategies,
                'hardware_targets': hardware_targets
            }
        }), 200
        
    except Exception as e:
        logger.error(f"获取优化技术失败: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'获取优化技术失败: {str(e)}'
        }), 500


def _get_technique_description(technique: str) -> str:
    """获取优化技术描述"""
    descriptions = {
        'quantization': '量化压缩，将模型权重从高精度转换为低精度表示',
        'pruning': '剪枝压缩，移除模型中不重要的权重或神经元',
        'knowledge_distillation': '知识蒸馏，使用小模型学习大模型的知识',
        'low_rank_decomposition': '低秩分解，将权重矩阵分解为低秩矩阵'
    }
    return descriptions.get(technique, '未知技术')


def _get_strategy_description(strategy: str) -> str:
    """获取压缩策略描述"""
    descriptions = {
        'structured': '结构化剪枝，移除整个神经元或卷积核',
        'unstructured': '非结构化剪枝，移除单个权重',
        'mixed': '混合策略，结合结构化和非结构化方法'
    }
    return descriptions.get(strategy, '未知策略')


@model_optimization_bp.route('/models/<model_id>/optimization-history', methods=['GET'])
@jwt_required()
def get_optimization_history(model_id: str):
    """
    获取模型优化历史
    
    Args:
        model_id: 模型ID
        
    Query Parameters:
        optimization_type: 优化类型过滤 (compression, inference, auto)
        status: 状态过滤 (pending, running, completed, failed)
        limit: 限制返回记录数，默认10
        
    Returns:
        优化历史记录
    """
    try:
        user_id = get_jwt_identity()
        tenant_id = request.headers.get('X-Tenant-ID') or request.args.get('tenant_id')
        
        optimization_type = request.args.get('optimization_type')
        status = request.args.get('status')
        limit = request.args.get('limit', 10, type=int)
        
        logger.info(f"用户 {user_id} 请求获取模型 {model_id} 的优化历史")
        
        history = optimization_service.get_optimization_history(
            model_id=model_id,
            tenant_id=tenant_id,
            optimization_type=optimization_type,
            status=status,
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
        logger.error(f"获取优化历史失败: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'获取优化历史失败: {str(e)}'
        }), 500


@model_optimization_bp.route('/optimization/<optimization_id>', methods=['GET'])
@jwt_required()
def get_optimization_detail(optimization_id: str):
    """
    获取单个优化记录详情
    
    Args:
        optimization_id: 优化ID
        
    Returns:
        优化详情
    """
    try:
        user_id = get_jwt_identity()
        
        logger.info(f"用户 {user_id} 请求获取优化详情: {optimization_id}")
        
        optimization = optimization_service.get_optimization_by_id(optimization_id)
        
        if not optimization:
            return jsonify({
                'success': False,
                'error': f'优化记录不存在: {optimization_id}'
            }), 404
        
        return jsonify({
            'success': True,
            'data': optimization
        }), 200
        
    except Exception as e:
        logger.error(f"获取优化详情失败: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'获取优化详情失败: {str(e)}'
        }), 500


@model_optimization_bp.route('/optimization/<optimization_id>', methods=['DELETE'])
@jwt_required()
def delete_optimization(optimization_id: str):
    """
    删除优化记录
    
    Args:
        optimization_id: 优化ID
        
    Returns:
        删除结果
    """
    try:
        user_id = get_jwt_identity()
        tenant_id = request.headers.get('X-Tenant-ID') or request.args.get('tenant_id')
        
        logger.info(f"用户 {user_id} 请求删除优化记录: {optimization_id}")
        
        success = optimization_service.delete_optimization(
            optimization_id=optimization_id,
            tenant_id=tenant_id
        )
        
        if success:
            return jsonify({
                'success': True,
                'message': f'优化记录已删除: {optimization_id}'
            }), 200
        else:
            return jsonify({
                'success': False,
                'error': f'删除优化记录失败: {optimization_id}'
            }), 404
            
    except Exception as e:
        logger.error(f"删除优化记录失败: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'删除优化记录失败: {str(e)}'
        }), 500


@model_optimization_bp.route('/optimizations', methods=['GET'])
@jwt_required()
def get_user_optimizations():
    """
    获取用户的优化记录列表
    
    Query Parameters:
        optimization_type: 优化类型过滤
        technique: 技术过滤
        status: 状态过滤
        limit: 返回数量限制，默认100
        offset: 偏移量，默认0
        
    Returns:
        优化记录列表
    """
    try:
        user_id = get_jwt_identity()
        tenant_id = request.headers.get('X-Tenant-ID') or request.args.get('tenant_id')
        
        optimization_type = request.args.get('optimization_type')
        technique = request.args.get('technique')
        status = request.args.get('status')
        limit = request.args.get('limit', 100, type=int)
        offset = request.args.get('offset', 0, type=int)
        
        logger.info(f"用户 {user_id} 请求获取优化记录列表")
        
        result = optimization_service.get_user_optimizations(
            user_id=user_id,
            tenant_id=tenant_id,
            optimization_type=optimization_type,
            technique=technique,
            status=status,
            limit=limit,
            offset=offset
        )
        
        return jsonify({
            'success': True,
            'data': result
        }), 200
        
    except Exception as e:
        logger.error(f"获取优化记录列表失败: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'获取优化记录列表失败: {str(e)}'
        }), 500


@model_optimization_bp.route('/statistics', methods=['GET'])
@jwt_required()
def get_optimization_statistics():
    """
    获取优化统计信息
    
    Query Parameters:
        model_id: 模型ID（可选）
        
    Returns:
        统计信息
    """
    try:
        user_id = get_jwt_identity()
        tenant_id = request.headers.get('X-Tenant-ID') or request.args.get('tenant_id')
        
        if not tenant_id:
            return jsonify({
                'success': False,
                'error': '获取优化统计需要提供租户ID'
            }), 400
        
        model_id = request.args.get('model_id')
        
        logger.info(f"用户 {user_id} 请求获取优化统计, tenant_id={tenant_id}")
        
        stats = optimization_service.get_optimization_statistics(
            tenant_id=tenant_id,
            user_id=user_id,
            model_id=model_id
        )
        
        return jsonify({
            'success': True,
            'data': stats
        }), 200
        
    except Exception as e:
        logger.error(f"获取优化统计失败: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'获取优化统计失败: {str(e)}'
        }), 500


@model_optimization_bp.route('/models/<model_id>/best', methods=['GET'])
@jwt_required()
def get_best_optimization(model_id: str):
    """
    获取模型的最佳优化记录
    
    Args:
        model_id: 模型ID
        
    Query Parameters:
        optimization_type: 优化类型过滤
        metric: 评估指标 (inference_speedup, model_size_reduction, accuracy_preserved)
        
    Returns:
        最佳优化记录
    """
    try:
        user_id = get_jwt_identity()
        tenant_id = request.headers.get('X-Tenant-ID') or request.args.get('tenant_id')
        
        optimization_type = request.args.get('optimization_type')
        metric = request.args.get('metric', 'inference_speedup')
        
        logger.info(f"用户 {user_id} 请求获取模型 {model_id} 的最佳优化")
        
        best = optimization_service.get_best_optimization(
            model_id=model_id,
            tenant_id=tenant_id,
            optimization_type=optimization_type,
            metric=metric
        )
        
        if best:
            return jsonify({
                'success': True,
                'data': best
            }), 200
        else:
            return jsonify({
                'success': False,
                'error': f'未找到模型 {model_id} 的优化记录'
            }), 404
            
    except Exception as e:
        logger.error(f"获取最佳优化失败: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'获取最佳优化失败: {str(e)}'
        }), 500


@model_optimization_bp.route('/compare', methods=['POST'])
@jwt_required()
def compare_optimizations():
    """
    比较多个优化记录
    
    Request Body:
        optimization_ids: 优化ID列表（至少2个）
        
    Returns:
        比较结果
    """
    try:
        user_id = get_jwt_identity()
        tenant_id = request.headers.get('X-Tenant-ID') or request.args.get('tenant_id')
        
        data = request.get_json() or {}
        optimization_ids = data.get('optimization_ids', [])
        
        if len(optimization_ids) < 2:
            return jsonify({
                'success': False,
                'error': '需要至少两个优化记录进行比较'
            }), 400
        
        logger.info(f"用户 {user_id} 请求比较 {len(optimization_ids)} 个优化记录")
        
        result = optimization_service.compare_optimizations(
            optimization_ids=optimization_ids,
            tenant_id=tenant_id
        )
        
        if 'error' in result and result.get('optimizations', []) == []:
            return jsonify({
                'success': False,
                'error': result['error']
            }), 400
        
        return jsonify({
            'success': True,
            'data': result
        }), 200
        
    except Exception as e:
        logger.error(f"比较优化记录失败: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'比较优化记录失败: {str(e)}'
        }), 500

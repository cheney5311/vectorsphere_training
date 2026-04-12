"""资源优化管理API接口

提供资源优化管理相关的API接口，实现生产级的资源优化功能：
- 优化状态管理（启动、停止、查询）
- 优化建议管理（获取、应用、忽略）
- 资源指标查询（当前值、历史数据）
- 性能分析和瓶颈检测
- 资源告警管理
"""

import sys
import os
from flask import Blueprint, request, jsonify, g
from flask_jwt_extended import jwt_required, get_jwt_identity
from datetime import datetime
import logging

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.core.exceptions import ValidationError
from backend.utils.response import success_response, error_response

logger = logging.getLogger(__name__)

# 创建蓝图
optimization_management_bp = Blueprint('optimization_management', __name__, url_prefix='/api/v1/optimization-management')


# ==================== 服务获取 ====================

def get_service():
    """获取优化管理服务实例"""
    try:
        from backend.services.optimization_management_service import get_optimization_management_service
        return get_optimization_management_service(use_memory=True)
    except ImportError as e:
        logger.error(f"Failed to import optimization service: {e}")
        return None


# ==================== 优化状态管理 ====================

@optimization_management_bp.route('/status', methods=['GET'])
@jwt_required()
def get_optimization_status():
    """获取资源优化状态
    
    获取当前优化运行状态、资源使用情况和统计信息。
    
    Returns:
        {
            "status": {
                "is_running": boolean,
                "current_session_id": string,
                "strategy": string,
                "progress": float,
                "started_at": string,
                "last_update": string,
                "active_optimizations": [],
                "resource_usage": {
                    "cpu": float,
                    "memory": float,
                    "gpu": float,
                    "disk": float
                },
                "statistics": {
                    "pending_recommendations": int,
                    "active_alerts": {}
                }
            }
        }
    """
    try:
        user_id = get_jwt_identity()
        tenant_id = getattr(g, 'tenant_id', None)
        
        service = get_service()
        if not service:
            return error_response("Optimization service not available", 503)
        
        status = service.get_status(tenant_id=tenant_id)
        
        return success_response({
            'status': status
        }, "获取资源优化状态成功")
        
    except Exception as e:
        logger.error(f"Get optimization status failed: {e}", exc_info=True)
        return error_response(f"获取资源优化状态失败: {str(e)}", 500)


@optimization_management_bp.route('/start', methods=['POST'])
@jwt_required()
def start_optimization():
    """启动资源优化
    
    启动一个新的优化会话，系统将分析资源使用情况并生成优化建议。
    
    Request Body:
        {
            "strategy": string,  // 优化策略: balanced, performance, energy, cost, custom
            "target_resources": [string],  // 目标资源: ["cpu", "memory", "gpu", "disk"]
            "config": {}  // 自定义配置（可选）
        }
    
    Returns:
        {
            "success": boolean,
            "message": string,
            "session": {}
        }
    """
    try:
        user_id = get_jwt_identity()
        tenant_id = getattr(g, 'tenant_id', None)
        
        service = get_service()
        if not service:
            return error_response("Optimization service not available", 503)
        
        data = request.get_json() or {}
        strategy = data.get('strategy', 'balanced')
        target_resources = data.get('target_resources')
        config = data.get('config')
        
        # 验证策略
        valid_strategies = ['balanced', 'performance', 'energy', 'cost', 'custom']
        if strategy not in valid_strategies:
            return error_response(
                f"无效的优化策略，支持: {', '.join(valid_strategies)}", 400
            )
        
        # 验证目标资源
        if target_resources:
            valid_resources = ['cpu', 'memory', 'gpu', 'disk', 'network']
            invalid = [r for r in target_resources if r not in valid_resources]
            if invalid:
                return error_response(
                    f"无效的目标资源: {', '.join(invalid)}，支持: {', '.join(valid_resources)}", 400
                )
        
        result = service.start_optimization(
            strategy=strategy,
            target_resources=target_resources,
            tenant_id=tenant_id,
            user_id=user_id,
            config=config
        )
        
        if result.get('success'):
            return success_response(result, f"资源优化已启动，使用策略: {strategy}")
        else:
            return error_response(message=result.get('message', '启动失败'), code=400, details=result.get('session'))
        
    except ValueError as e:
        return error_response(str(e), 400)
    except Exception as e:
        logger.error(f"Start optimization failed: {e}", exc_info=True)
        return error_response(f"启动资源优化失败: {str(e)}", 500)


@optimization_management_bp.route('/stop', methods=['POST'])
@jwt_required()
def stop_optimization():
    """停止资源优化
    
    停止当前运行的优化会话。
    
    Returns:
        {
            "success": boolean,
            "message": string,
            "session_id": string
        }
    """
    try:
        user_id = get_jwt_identity()
        tenant_id = getattr(g, 'tenant_id', None)
        
        service = get_service()
        if not service:
            return error_response("Optimization service not available", 503)
        
        result = service.stop_optimization(
            tenant_id=tenant_id,
            user_id=user_id
        )
        
        return success_response(result, result.get('message', '资源优化已停止'))
        
    except Exception as e:
        logger.error(f"Stop optimization failed: {e}", exc_info=True)
        return error_response(f"停止资源优化失败: {str(e)}", 500)


# ==================== 优化建议管理 ====================

@optimization_management_bp.route('/recommendations', methods=['GET'])
@jwt_required()
def get_recommendations():
    """获取优化建议列表
    
    Query Parameters:
        - category: 资源类别过滤 (cpu/memory/gpu/disk)
        - status: 状态过滤 (pending/applied/ignored/failed/expired)
        - priority: 优先级过滤 (low/medium/high/critical)
        - limit: 返回数量限制 (默认: 100)
        - offset: 偏移量 (默认: 0)
    
    Returns:
        {
            "recommendations": [
                {
                    "id": string,
                    "title": string,
                    "description": string,
                    "category": string,
                    "priority": string,
                    "confidence": float,
                    "status": string,
                    "action": string,
                    "estimated_impact": string,
                    "estimated_savings_percent": float,
                    "risk_level": string,
                    "current_value": float,
                    "recommended_value": float,
                    "created_at": string
                }
            ],
            "total": int,
            "limit": int,
            "offset": int
        }
    """
    try:
        user_id = get_jwt_identity()
        tenant_id = getattr(g, 'tenant_id', None)
        
        service = get_service()
        if not service:
            return error_response("Optimization service not available", 503)
        
        # 获取查询参数
        category = request.args.get('category')
        status = request.args.get('status')
        priority = request.args.get('priority')
        limit = min(int(request.args.get('limit', 100)), 1000)
        offset = int(request.args.get('offset', 0))
        
        result = service.get_recommendations(
            tenant_id=tenant_id,
            category=category,
            status=status,
            priority=priority,
            limit=limit,
            offset=offset
        )
        
        return success_response(result, "获取优化建议成功")
        
    except Exception as e:
        logger.error(f"Get recommendations failed: {e}", exc_info=True)
        return error_response(f"获取优化建议失败: {str(e)}", 500)


@optimization_management_bp.route('/recommendations/<recommendation_id>/apply', methods=['POST'])
@jwt_required()
def apply_recommendation(recommendation_id):
    """应用优化建议
    
    执行指定的优化建议操作。
    
    Path Parameters:
        - recommendation_id: 建议ID
    
    Returns:
        {
            "success": boolean,
            "message": string,
            "result": {}
        }
    """
    try:
        user_id = get_jwt_identity()
        tenant_id = getattr(g, 'tenant_id', None)
        
        service = get_service()
        if not service:
            return error_response("Optimization service not available", 503)
        
        result = service.apply_recommendation(
            recommendation_id=recommendation_id,
            user_id=user_id,
            tenant_id=tenant_id
        )
        
        if result.get('success'):
            return success_response(result, result.get('message', '优化建议已应用'))
        else:
            return error_response(result.get('message', '应用失败'), 400)
        
    except Exception as e:
        logger.error(f"Apply recommendation failed: {e}", exc_info=True)
        return error_response(f"应用优化建议失败: {str(e)}", 500)


@optimization_management_bp.route('/recommendations/<recommendation_id>/ignore', methods=['POST'])
@jwt_required()
def ignore_recommendation(recommendation_id):
    """忽略优化建议
    
    将指定的建议标记为已忽略。
    
    Path Parameters:
        - recommendation_id: 建议ID
    
    Returns:
        {
            "success": boolean,
            "message": string
        }
    """
    try:
        user_id = get_jwt_identity()
        
        service = get_service()
        if not service:
            return error_response("Optimization service not available", 503)
        
        result = service.ignore_recommendation(
            recommendation_id=recommendation_id,
            user_id=user_id
        )
        
        if result.get('success'):
            return success_response(result, '优化建议已忽略')
        else:
            return error_response(result.get('message', '忽略失败'), 400)
        
    except Exception as e:
        logger.error(f"Ignore recommendation failed: {e}", exc_info=True)
        return error_response(f"忽略优化建议失败: {str(e)}", 500)


# ==================== 资源指标管理 ====================

@optimization_management_bp.route('/metrics', methods=['GET'])
@jwt_required()
def get_current_metrics():
    """获取当前资源指标
    
    获取实时的资源使用情况指标。
    
    Returns:
        {
            "timestamp": string,
            "cpu": {
                "utilization": float,
                "load_avg": float,
                "cores": int
            },
            "memory": {
                "utilization": float,
                "available_mb": float,
                "total_mb": float
            },
            "gpu": {
                "utilization": float,
                "memory_used_mb": float,
                "temperature": float
            },
            "disk": {
                "utilization": float,
                "free_gb": float
            },
            "network": {
                "bytes_sent": int,
                "bytes_recv": int
            },
            "status": {
                "cpu": string,
                "memory": string,
                "gpu": string,
                "disk": string
            },
            "overall_status": string
        }
    """
    try:
        user_id = get_jwt_identity()
        tenant_id = getattr(g, 'tenant_id', None)
        
        service = get_service()
        if not service:
            return error_response("Optimization service not available", 503)
        
        metrics = service.get_current_metrics(tenant_id=tenant_id)
        
        return success_response(metrics, "获取资源指标成功")
        
    except Exception as e:
        logger.error(f"Get current metrics failed: {e}", exc_info=True)
        return error_response(f"获取资源指标失败: {str(e)}", 500)


@optimization_management_bp.route('/metrics/history', methods=['GET'])
@jwt_required()
def get_metrics_history():
    """获取资源指标历史数据
    
    Query Parameters:
        - type: 指标类型 (cpu, memory, gpu, disk, network)
        - hours: 获取多少小时的数据 (默认: 1，最大: 168)
        - resolution: 分辨率 (minute, hour, day)
    
    Returns:
        {
            "history": [
                {
                    "timestamp": string,
                    "value": float,
                    "type": string
                }
            ],
            "count": int,
            "metric_type": string,
            "resolution": string,
            "time_range": {
                "start": string,
                "end": string
            }
        }
    """
    try:
        user_id = get_jwt_identity()
        tenant_id = getattr(g, 'tenant_id', None)
        
        service = get_service()
        if not service:
            return error_response("Optimization service not available", 503)
        
        # 获取查询参数
        metric_type = request.args.get('type', 'cpu')
        hours = int(request.args.get('hours', 1))
        resolution = request.args.get('resolution', 'minute')
        
        # 验证参数
        if hours <= 0 or hours > 168:
            return error_response("hours参数必须在1-168之间", 400)
        
        valid_types = ['cpu', 'memory', 'gpu', 'disk', 'network']
        if metric_type not in valid_types:
            return error_response(f"无效的指标类型，支持: {', '.join(valid_types)}", 400)
        
        valid_resolutions = ['minute', 'hour', 'day']
        if resolution not in valid_resolutions:
            return error_response(f"无效的分辨率，支持: {', '.join(valid_resolutions)}", 400)
        
        result = service.get_metrics_history(
            metric_type=metric_type,
            tenant_id=tenant_id,
            hours=hours,
            resolution=resolution
        )
        
        return success_response(result, "获取资源指标历史数据成功")
        
    except ValueError as e:
        return error_response(f"参数错误: {str(e)}", 400)
    except Exception as e:
        logger.error(f"Get metrics history failed: {e}", exc_info=True)
        return error_response(f"获取资源指标历史数据失败: {str(e)}", 500)


# ==================== 性能分析 ====================

@optimization_management_bp.route('/analyze', methods=['POST'])
@jwt_required()
def analyze_performance():
    """执行性能分析
    
    对系统进行性能分析，检测瓶颈并生成优化建议。
    
    Request Body:
        {
            "type": string,  // 分析类型: full, cpu, memory, io
            "target_id": string  // 可选，特定任务或进程ID
        }
    
    Returns:
        {
            "report_id": string,
            "timestamp": string,
            "analysis_type": string,
            "summary": string,
            "bottlenecks": [
                {
                    "type": string,
                    "severity": string,
                    "description": string,
                    "metrics": {}
                }
            ],
            "recommendations": [
                {
                    "title": string,
                    "description": string,
                    "priority": string,
                    "estimated_impact": string
                }
            ],
            "metrics_summary": {}
        }
    """
    try:
        user_id = get_jwt_identity()
        tenant_id = getattr(g, 'tenant_id', None)
        
        service = get_service()
        if not service:
            return error_response("Optimization service not available", 503)
        
        data = request.get_json() or {}
        analysis_type = data.get('type', 'full')
        target_id = data.get('target_id')
        
        # 验证分析类型
        valid_types = ['full', 'cpu', 'memory', 'io']
        if analysis_type not in valid_types:
            return error_response(f"无效的分析类型，支持: {', '.join(valid_types)}", 400)
        
        result = service.analyze_performance(
            analysis_type=analysis_type,
            target_id=target_id,
            tenant_id=tenant_id,
            user_id=user_id
        )
        
        return success_response(result, "性能分析完成")
        
    except Exception as e:
        logger.error(f"Performance analysis failed: {e}", exc_info=True)
        return error_response(f"执行性能分析失败: {str(e)}", 500)


# ==================== 告警管理 ====================

@optimization_management_bp.route('/alerts', methods=['GET'])
@jwt_required()
def get_alerts():
    """获取资源告警列表
    
    Query Parameters:
        - level: 级别过滤 (info, warning, critical)
        - resource_type: 资源类型过滤 (cpu, memory, gpu, disk)
        - status: 状态过滤 (active, acknowledged, resolved)
        - limit: 返回数量限制
        - offset: 偏移量
    
    Returns:
        {
            "alerts": [
                {
                    "id": string,
                    "level": string,
                    "resource_type": string,
                    "message": string,
                    "metric_value": float,
                    "threshold": float,
                    "status": string,
                    "timestamp": string
                }
            ],
            "total": int
        }
    """
    try:
        user_id = get_jwt_identity()
        tenant_id = getattr(g, 'tenant_id', None)
        
        service = get_service()
        if not service:
            return error_response("Optimization service not available", 503)
        
        # 获取查询参数
        level = request.args.get('level')
        resource_type = request.args.get('resource_type')
        status = request.args.get('status')
        limit = min(int(request.args.get('limit', 100)), 1000)
        offset = int(request.args.get('offset', 0))
        
        result = service.get_alerts(
            tenant_id=tenant_id,
            level=level,
            resource_type=resource_type,
            status=status,
            limit=limit,
            offset=offset
        )
        
        return success_response(result, "获取资源告警成功")
        
    except Exception as e:
        logger.error(f"Get alerts failed: {e}", exc_info=True)
        return error_response(f"获取资源告警失败: {str(e)}", 500)


@optimization_management_bp.route('/alerts/<alert_id>/acknowledge', methods=['POST'])
@jwt_required()
def acknowledge_alert(alert_id):
    """确认告警
    
    将告警标记为已确认状态。
    
    Path Parameters:
        - alert_id: 告警ID
    
    Returns:
        {
            "success": boolean,
            "alert": {}
        }
    """
    try:
        user_id = get_jwt_identity()
        
        service = get_service()
        if not service:
            return error_response("Optimization service not available", 503)
        
        result = service.acknowledge_alert(
            alert_id=alert_id,
            user_id=user_id
        )
        
        if result.get('success'):
            return success_response(result, '告警已确认')
        else:
            return error_response(result.get('message', '确认失败'), 404)
        
    except Exception as e:
        logger.error(f"Acknowledge alert failed: {e}", exc_info=True)
        return error_response(f"确认告警失败: {str(e)}", 500)


@optimization_management_bp.route('/alerts/<alert_id>/resolve', methods=['POST'])
@jwt_required()
def resolve_alert(alert_id):
    """解决告警
    
    将告警标记为已解决状态。
    
    Path Parameters:
        - alert_id: 告警ID
    
    Request Body (optional):
        {
            "note": string  // 解决备注
        }
    
    Returns:
        {
            "success": boolean,
            "alert": {}
        }
    """
    try:
        user_id = get_jwt_identity()
        
        service = get_service()
        if not service:
            return error_response("Optimization service not available", 503)
        
        data = request.get_json() or {}
        note = data.get('note')
        
        result = service.resolve_alert(
            alert_id=alert_id,
            user_id=user_id,
            note=note
        )
        
        if result.get('success'):
            return success_response(result, '告警已解决')
        else:
            return error_response(result.get('message', '解决失败'), 404)
        
    except Exception as e:
        logger.error(f"Resolve alert failed: {e}", exc_info=True)
        return error_response(f"解决告警失败: {str(e)}", 500)


@optimization_management_bp.route('/alerts/statistics', methods=['GET'])
@jwt_required()
def get_alert_statistics():
    """获取告警统计
    
    Returns:
        {
            "total": int,
            "critical": int,
            "warning": int,
            "info": int
        }
    """
    try:
        user_id = get_jwt_identity()
        tenant_id = getattr(g, 'tenant_id', None)
        
        service = get_service()
        if not service:
            return error_response("Optimization service not available", 503)
        
        statistics = service.get_alert_statistics(tenant_id=tenant_id)
        
        return success_response(statistics, "获取告警统计成功")
        
    except Exception as e:
        logger.error(f"Get alert statistics failed: {e}", exc_info=True)
        return error_response(f"获取告警统计失败: {str(e)}", 500)


# ==================== 会话管理 ====================

@optimization_management_bp.route('/sessions', methods=['GET'])
@jwt_required()
def list_sessions():
    """获取优化会话列表
    
    Query Parameters:
        - status: 状态过滤 (idle, analyzing, optimizing, completed, failed, cancelled)
        - limit: 返回数量限制
        - offset: 偏移量
    
    Returns:
        {
            "sessions": [
                {
                    "id": string,
                    "name": string,
                    "strategy": string,
                    "status": string,
                    "progress": float,
                    "recommendations_count": int,
                    "applied_count": int,
                    "estimated_savings": float,
                    "created_at": string,
                    "completed_at": string
                }
            ],
            "total": int
        }
    """
    try:
        user_id = get_jwt_identity()
        tenant_id = getattr(g, 'tenant_id', None)
        
        service = get_service()
        if not service:
            return error_response("Optimization service not available", 503)
        
        status = request.args.get('status')
        limit = min(int(request.args.get('limit', 50)), 500)
        offset = int(request.args.get('offset', 0))
        
        result = service.list_sessions(
            tenant_id=tenant_id,
            status=status,
            limit=limit,
            offset=offset
        )
        
        return success_response(result, "获取优化会话列表成功")
        
    except Exception as e:
        logger.error(f"List sessions failed: {e}", exc_info=True)
        return error_response(f"获取优化会话列表失败: {str(e)}", 500)


@optimization_management_bp.route('/sessions/<session_id>', methods=['GET'])
@jwt_required()
def get_session(session_id):
    """获取优化会话详情
    
    Path Parameters:
        - session_id: 会话ID
    
    Returns:
        {
            "session": {}
        }
    """
    try:
        user_id = get_jwt_identity()
        
        service = get_service()
        if not service:
            return error_response("Optimization service not available", 503)
        
        session = service.get_session(session_id)
        
        if session:
            return success_response({'session': session}, "获取会话详情成功")
        else:
            return error_response("会话不存在", 404)
        
    except Exception as e:
        logger.error(f"Get session failed: {e}", exc_info=True)
        return error_response(f"获取会话详情失败: {str(e)}", 500)


# ==================== 训练优化 ====================

@optimization_management_bp.route('/training/optimize', methods=['POST'])
@jwt_required()
def optimize_training():
    """执行训练优化
    
    对指定的训练任务执行资源优化，包括图优化、内存优化、资源调度等。
    
    Request Body:
        {
            "training_job_id": string,      // 必需：训练任务ID
            "model_id": string,             // 可选：模型ID
            "optimization_types": [string], // 可选：优化类型列表
            "config": {}                    // 可选：优化配置
        }
    
    Returns:
        {
            "success": boolean,
            "message": string,
            "result": {
                "optimization_id": string,
                "training_job_id": string,
                "optimizations_applied": [string],
                "total_performance_improvement": float,
                "total_memory_reduction": float,
                "execution_time_seconds": float,
                "resource_impact": {},
                "details": {}
            }
        }
    """
    try:
        user_id = get_jwt_identity()
        tenant_id = getattr(g, 'tenant_id', None)
        
        service = get_service()
        if not service:
            return error_response("Optimization service not available", 503)
        
        data = request.get_json()
        if not data:
            return error_response("Request body is required", 400)
        
        training_job_id = data.get('training_job_id')
        if not training_job_id:
            return error_response("training_job_id is required", 400)
        
        model_id = data.get('model_id')
        optimization_types = data.get('optimization_types')
        config = data.get('config')
        
        # 验证优化类型
        if optimization_types:
            valid_types = [
                'graph_optimization', 'memory_optimization', 'operator_fusion',
                'constant_folding', 'dead_code_elimination', 'layout_optimization',
                'resource_scheduling', 'batch_optimization', 'mixed_precision',
                'gradient_accumulation'
            ]
            invalid = [t for t in optimization_types if t not in valid_types]
            if invalid:
                return error_response(
                    f"无效的优化类型: {', '.join(invalid)}",
                    400,
                    details={'valid_types': valid_types}
                )
        
        result = service.optimize_training(
            training_job_id=training_job_id,
            model_id=model_id,
            optimization_types=optimization_types,
            config=config,
            tenant_id=tenant_id,
            user_id=user_id
        )
        
        if result.get('success'):
            return success_response(result, result.get('message', '训练优化完成'))
        else:
            return error_response(result.get('message', '训练优化失败'), 500)
        
    except Exception as e:
        logger.error(f"Training optimization failed: {e}", exc_info=True)
        return error_response(f"执行训练优化失败: {str(e)}", 500)


@optimization_management_bp.route('/training/optimization-types', methods=['GET'])
@jwt_required()
def get_optimization_types():
    """获取可用的优化类型列表
    
    Returns:
        {
            "optimization_types": [
                {
                    "type": string,
                    "name": string,
                    "description": string,
                    "category": string,
                    "estimated_improvement": string
                }
            ]
        }
    """
    try:
        service = get_service()
        if not service:
            return error_response("Optimization service not available", 503)
        
        types = service.get_available_optimization_types()
        
        return success_response({
            'optimization_types': types,
            'count': len(types)
        }, "获取优化类型列表成功")
        
    except Exception as e:
        logger.error(f"Get optimization types failed: {e}", exc_info=True)
        return error_response(f"获取优化类型列表失败: {str(e)}", 500)


@optimization_management_bp.route('/training/history', methods=['GET'])
@jwt_required()
def get_training_optimization_history():
    """获取训练优化历史记录
    
    Query Parameters:
        - training_job_id: 训练任务ID（可选）
        - limit: 返回数量限制
        - offset: 偏移量
    
    Returns:
        {
            "optimizations": [],
            "total": int
        }
    """
    try:
        user_id = get_jwt_identity()
        tenant_id = getattr(g, 'tenant_id', None)
        
        service = get_service()
        if not service:
            return error_response("Optimization service not available", 503)
        
        training_job_id = request.args.get('training_job_id')
        limit = min(int(request.args.get('limit', 50)), 500)
        offset = int(request.args.get('offset', 0))
        
        result = service.get_training_optimization_history(
            training_job_id=training_job_id,
            tenant_id=tenant_id,
            limit=limit,
            offset=offset
        )
        
        return success_response(result, "获取训练优化历史成功")
        
    except Exception as e:
        logger.error(f"Get training optimization history failed: {e}", exc_info=True)
        return error_response(f"获取训练优化历史失败: {str(e)}", 500)


@optimization_management_bp.route('/training/<training_job_id>/quick-optimize', methods=['POST'])
@jwt_required()
def quick_optimize_training(training_job_id):
    """快速训练优化
    
    使用默认配置快速执行常用优化。
    
    Path Parameters:
        - training_job_id: 训练任务ID
    
    Returns:
        {
            "success": boolean,
            "message": string,
            "result": {}
        }
    """
    try:
        user_id = get_jwt_identity()
        tenant_id = getattr(g, 'tenant_id', None)
        
        service = get_service()
        if not service:
            return error_response("Optimization service not available", 503)
        
        # 使用常用优化组合
        default_optimizations = [
            'graph_optimization',
            'memory_optimization',
            'resource_scheduling'
        ]
        
        result = service.optimize_training(
            training_job_id=training_job_id,
            optimization_types=default_optimizations,
            config={'strategy': 'balanced'},
            tenant_id=tenant_id,
            user_id=user_id
        )
        
        if result.get('success'):
            return success_response(result, '快速优化完成')
        else:
            return error_response(result.get('message', '快速优化失败'), 500)
        
    except Exception as e:
        logger.error(f"Quick optimize training failed: {e}", exc_info=True)
        return error_response(f"快速优化失败: {str(e)}", 500)


@optimization_management_bp.route('/training/<training_job_id>/graph-optimize', methods=['POST'])
@jwt_required()
def graph_optimize_training(training_job_id):
    """执行图优化
    
    对训练任务的模型图执行优化。
    
    Path Parameters:
        - training_job_id: 训练任务ID
    
    Request Body (optional):
        {
            "model_id": string,
            "optimizations": [string]  // constant_folding, dead_code_elimination, operator_fusion, layout_optimization
        }
    
    Returns:
        {
            "success": boolean,
            "message": string,
            "result": {}
        }
    """
    try:
        user_id = get_jwt_identity()
        tenant_id = getattr(g, 'tenant_id', None)
        
        service = get_service()
        if not service:
            return error_response("Optimization service not available", 503)
        
        data = request.get_json() or {}
        model_id = data.get('model_id')
        graph_optimizations = data.get('optimizations')
        
        result = service.optimize_training(
            training_job_id=training_job_id,
            model_id=model_id,
            optimization_types=['graph_optimization'],
            config={'graph_optimizations': graph_optimizations},
            tenant_id=tenant_id,
            user_id=user_id
        )
        
        if result.get('success'):
            return success_response(result, '图优化完成')
        else:
            return error_response(result.get('message', '图优化失败'), 500)
        
    except Exception as e:
        logger.error(f"Graph optimize training failed: {e}", exc_info=True)
        return error_response(f"图优化失败: {str(e)}", 500)


@optimization_management_bp.route('/training/<training_job_id>/resource-optimize', methods=['POST'])
@jwt_required()
def resource_optimize_training(training_job_id):
    """执行资源优化
    
    优化训练任务的资源使用。
    
    Path Parameters:
        - training_job_id: 训练任务ID
    
    Request Body (optional):
        {
            "cpu_target": float,     // CPU目标利用率
            "memory_target": float,  // 内存目标利用率
            "gpu_target": float      // GPU目标利用率
        }
    
    Returns:
        {
            "success": boolean,
            "message": string,
            "result": {}
        }
    """
    try:
        user_id = get_jwt_identity()
        tenant_id = getattr(g, 'tenant_id', None)
        
        service = get_service()
        if not service:
            return error_response("Optimization service not available", 503)
        
        data = request.get_json() or {}
        
        result = service.optimize_training(
            training_job_id=training_job_id,
            optimization_types=['resource_scheduling', 'memory_optimization'],
            config={
                'cpu_target': data.get('cpu_target', 70),
                'memory_target': data.get('memory_target', 75),
                'gpu_target': data.get('gpu_target', 80)
            },
            tenant_id=tenant_id,
            user_id=user_id
        )
        
        if result.get('success'):
            return success_response(result, '资源优化完成')
        else:
            return error_response(result.get('message', '资源优化失败'), 500)
        
    except Exception as e:
        logger.error(f"Resource optimize training failed: {e}", exc_info=True)
        return error_response(f"资源优化失败: {str(e)}", 500)


# ==================== 导出 ====================

__all__ = ['optimization_management_bp']

"""资源优化统计API接口

提供资源优化统计相关的API接口，实现生产级的资源监控功能：
- 实时资源指标获取
- 历史指标数据查询
- 资源告警查询
- 性能分析执行
"""

import sys
import os
from flask import Blueprint, request, jsonify, g
from flask_jwt_extended import jwt_required, get_jwt_identity
from datetime import datetime, timedelta
import logging

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.core.exceptions import ValidationError
from backend.utils.response import success_response, error_response

logger = logging.getLogger(__name__)

# 创建蓝图
optimization_statistics_bp = Blueprint('optimization_statistics', __name__, url_prefix='/api/v1/optimization')


# ==================== 服务获取 ====================

def get_service():
    """获取优化管理服务实例"""
    try:
        from backend.services.optimization_management_service import get_optimization_management_service
        return get_optimization_management_service(use_memory=True)
    except ImportError as e:
        logger.error(f"Failed to import optimization service: {e}")
        return None


# ==================== 资源指标 ====================

@optimization_statistics_bp.route('/metrics', methods=['GET'])
@jwt_required()
def get_resource_metrics():
    """获取当前资源指标
    
    获取实时的系统资源使用情况，包括CPU、内存、GPU、磁盘和网络。
    
    Returns:
        {
            "timestamp": string,
            "cpu": {
                "utilization": float,
                "load_avg": float,
                "cores": int,
                "frequency": float
            },
            "memory": {
                "utilization": float,
                "available_mb": float,
                "total_mb": float,
                "used_mb": float
            },
            "gpu": {
                "utilization": float,
                "memory_used_mb": float,
                "memory_total_mb": float,
                "temperature": float,
                "power_usage": float
            },
            "disk": {
                "utilization": float,
                "free_gb": float,
                "total_gb": float
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
        logger.error(f"Get resource metrics failed: {e}", exc_info=True)
        return error_response(f"获取资源指标失败: {str(e)}", 500)


@optimization_statistics_bp.route('/metrics/history', methods=['GET'])
@jwt_required()
def get_metrics_history():
    """获取资源指标历史数据
    
    获取指定时间范围内的资源指标历史数据，支持不同的分辨率。
    
    Query Parameters:
        - type: 指标类型 (cpu, memory, gpu, disk, network)，默认: cpu
        - hours: 获取多少小时的数据 (1-168)，默认: 1
        - resolution: 分辨率 (minute, hour, day)，默认: minute
    
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
        
        # 获取并验证查询参数
        metric_type = request.args.get('type', 'cpu')
        
        try:
            hours = int(request.args.get('hours', 1))
        except (ValueError, TypeError):
            hours = 1
        
        resolution = request.args.get('resolution', 'minute')
        
        # 参数验证
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


@optimization_statistics_bp.route('/metrics/summary', methods=['GET'])
@jwt_required()
def get_metrics_summary():
    """获取资源指标汇总
    
    获取所有资源类型的当前状态汇总。
    
    Returns:
        {
            "summary": {
                "cpu": {
                    "current": float,
                    "status": string,
                    "trend": string
                },
                "memory": {...},
                "gpu": {...},
                "disk": {...}
            },
            "overall_status": string,
            "timestamp": string
        }
    """
    try:
        user_id = get_jwt_identity()
        tenant_id = getattr(g, 'tenant_id', None)
        
        service = get_service()
        if not service:
            return error_response("Optimization service not available", 503)
        
        metrics = service.get_current_metrics(tenant_id=tenant_id)
        
        summary = {
            'cpu': {
                'current': metrics.get('cpu', {}).get('utilization', 0),
                'status': metrics.get('status', {}).get('cpu', 'unknown'),
                'trend': 'stable'  # 可以通过历史数据计算
            },
            'memory': {
                'current': metrics.get('memory', {}).get('utilization', 0),
                'status': metrics.get('status', {}).get('memory', 'unknown'),
                'trend': 'stable'
            },
            'gpu': {
                'current': metrics.get('gpu', {}).get('utilization', 0),
                'status': metrics.get('status', {}).get('gpu', 'unknown'),
                'trend': 'stable'
            },
            'disk': {
                'current': metrics.get('disk', {}).get('utilization', 0),
                'status': metrics.get('status', {}).get('disk', 'unknown'),
                'trend': 'stable'
            }
        }
        
        return success_response({
            'summary': summary,
            'overall_status': metrics.get('overall_status', 'unknown'),
            'timestamp': metrics.get('timestamp', datetime.utcnow().isoformat())
        }, "获取资源指标汇总成功")
        
    except Exception as e:
        logger.error(f"Get metrics summary failed: {e}", exc_info=True)
        return error_response(f"获取资源指标汇总失败: {str(e)}", 500)


# ==================== 告警管理 ====================

@optimization_statistics_bp.route('/alerts', methods=['GET'])
@jwt_required()
def get_alerts():
    """获取资源告警列表
    
    获取当前活跃的资源告警，可以按级别和类型过滤。
    
    Query Parameters:
        - level: 级别过滤 (info, warning, critical)
        - resource_type: 资源类型过滤 (cpu, memory, gpu, disk, network)
        - status: 状态过滤 (active, acknowledged, resolved)
        - limit: 返回数量限制，默认: 100
        - offset: 偏移量，默认: 0
    
    Returns:
        {
            "alerts": [
                {
                    "id": string,
                    "timestamp": string,
                    "level": string,
                    "resource_type": string,
                    "message": string,
                    "metric_value": float,
                    "threshold": float,
                    "status": string
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
        
        try:
            limit = min(int(request.args.get('limit', 100)), 1000)
            offset = int(request.args.get('offset', 0))
        except (ValueError, TypeError):
            limit = 100
            offset = 0
        
        # 验证参数
        if level:
            valid_levels = ['info', 'warning', 'critical']
            if level not in valid_levels:
                return error_response(f"无效的级别，支持: {', '.join(valid_levels)}", 400)
        
        if resource_type:
            valid_types = ['cpu', 'memory', 'gpu', 'disk', 'network']
            if resource_type not in valid_types:
                return error_response(f"无效的资源类型，支持: {', '.join(valid_types)}", 400)
        
        if status:
            valid_statuses = ['active', 'acknowledged', 'resolved']
            if status not in valid_statuses:
                return error_response(f"无效的状态，支持: {', '.join(valid_statuses)}", 400)
        
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


@optimization_statistics_bp.route('/alerts/summary', methods=['GET'])
@jwt_required()
def get_alerts_summary():
    """获取告警汇总统计
    
    Returns:
        {
            "total": int,
            "by_level": {
                "critical": int,
                "warning": int,
                "info": int
            },
            "by_resource": {
                "cpu": int,
                "memory": int,
                "gpu": int,
                "disk": int
            }
        }
    """
    try:
        user_id = get_jwt_identity()
        tenant_id = getattr(g, 'tenant_id', None)
        
        service = get_service()
        if not service:
            return error_response("Optimization service not available", 503)
        
        statistics = service.get_alert_statistics(tenant_id=tenant_id)
        
        return success_response({
            'total': statistics.get('total', 0),
            'by_level': {
                'critical': statistics.get('critical', 0),
                'warning': statistics.get('warning', 0),
                'info': statistics.get('info', 0)
            }
        }, "获取告警汇总成功")
        
    except Exception as e:
        logger.error(f"Get alerts summary failed: {e}", exc_info=True)
        return error_response(f"获取告警汇总失败: {str(e)}", 500)


# ==================== 性能分析 ====================

@optimization_statistics_bp.route('/analyze', methods=['POST'])
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
                    "metrics": {
                        "avg_utilization": float,
                        "peak_utilization": float,
                        "recommended_limit": float
                    }
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


# ==================== 优化状态 ====================

@optimization_statistics_bp.route('/status', methods=['GET'])
@jwt_required()
def get_optimization_status():
    """获取优化状态概览
    
    Returns:
        {
            "is_running": boolean,
            "current_strategy": string,
            "progress": float,
            "resource_usage": {
                "cpu": float,
                "memory": float,
                "gpu": float,
                "disk": float
            },
            "pending_recommendations": int,
            "active_alerts": int
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
            'is_running': status.get('is_running', False),
            'current_strategy': status.get('strategy', 'balanced'),
            'progress': status.get('progress', 0),
            'resource_usage': status.get('resource_usage', {}),
            'pending_recommendations': status.get('statistics', {}).get('pending_recommendations', 0),
            'active_alerts': status.get('statistics', {}).get('active_alerts', {}).get('total', 0)
        }, "获取优化状态成功")
        
    except Exception as e:
        logger.error(f"Get optimization status failed: {e}", exc_info=True)
        return error_response(f"获取优化状态失败: {str(e)}", 500)


# ==================== 建议统计 ====================

@optimization_statistics_bp.route('/recommendations/statistics', methods=['GET'])
@jwt_required()
def get_recommendations_statistics():
    """获取优化建议统计
    
    Returns:
        {
            "total": int,
            "by_status": {
                "pending": int,
                "applied": int,
                "ignored": int,
                "failed": int
            },
            "by_category": {
                "cpu": int,
                "memory": int,
                "gpu": int,
                "disk": int
            },
            "by_priority": {
                "critical": int,
                "high": int,
                "medium": int,
                "low": int
            }
        }
    """
    try:
        user_id = get_jwt_identity()
        tenant_id = getattr(g, 'tenant_id', None)
        
        service = get_service()
        if not service:
            return error_response("Optimization service not available", 503)
        
        # 获取所有建议以计算统计
        all_recs = service.get_recommendations(tenant_id=tenant_id, limit=10000)
        recommendations = all_recs.get('recommendations', [])
        
        # 按状态统计
        by_status = {'pending': 0, 'applied': 0, 'ignored': 0, 'failed': 0, 'expired': 0}
        for rec in recommendations:
            status = rec.get('status', 'pending')
            if status in by_status:
                by_status[status] += 1
        
        # 按类别统计
        by_category = {'cpu': 0, 'memory': 0, 'gpu': 0, 'disk': 0, 'system': 0}
        for rec in recommendations:
            category = rec.get('category', 'system')
            if category in by_category:
                by_category[category] += 1
        
        # 按优先级统计
        by_priority = {'critical': 0, 'high': 0, 'medium': 0, 'low': 0}
        for rec in recommendations:
            priority = rec.get('priority', 'medium')
            if priority in by_priority:
                by_priority[priority] += 1
        
        return success_response({
            'total': len(recommendations),
            'by_status': by_status,
            'by_category': by_category,
            'by_priority': by_priority
        }, "获取建议统计成功")
        
    except Exception as e:
        logger.error(f"Get recommendations statistics failed: {e}", exc_info=True)
        return error_response(f"获取建议统计失败: {str(e)}", 500)


# ==================== 健康检查 ====================

@optimization_statistics_bp.route('/health', methods=['GET'])
def health_check():
    """健康检查端点
    
    不需要认证，用于负载均衡器健康检查。
    
    Returns:
        {
            "status": string,
            "timestamp": string,
            "service": string
        }
    """
    try:
        service = get_service()
        
        return jsonify({
            'success': True,
            'data': {
                'status': 'healthy' if service else 'degraded',
                'timestamp': datetime.utcnow().isoformat(),
                'service': 'optimization-statistics'
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            'success': False,
            'data': {
                'status': 'unhealthy',
                'timestamp': datetime.utcnow().isoformat(),
                'error': str(e)
            }
        }), 503


# ==================== 导出 ====================

__all__ = ['optimization_statistics_bp']

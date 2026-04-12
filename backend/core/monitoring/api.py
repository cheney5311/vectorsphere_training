"""统一监控API接口

提供REST API接口用于访问监控服务功能。
"""

from flask import Blueprint, jsonify, request
from typing import Dict, Any

from .service import get_monitoring_service
from .analyzer import get_performance_analyzer, AnalysisType
from .optimizer import get_resource_optimizer

# 创建蓝图
monitoring_bp = Blueprint('monitoring', __name__, url_prefix='/api/monitoring')


@monitoring_bp.route('/metrics', methods=['GET'])
def get_current_metrics():
    """获取当前系统指标"""
    try:
        service = get_monitoring_service()
        metrics = service.get_current_metrics()
        return jsonify({
            'success': True,
            'data': metrics
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@monitoring_bp.route('/metrics/history', methods=['GET'])
def get_metrics_history():
    """获取历史指标数据"""
    try:
        start_time = request.args.get('start_time', type=float)
        end_time = request.args.get('end_time', type=float)
        interval = request.args.get('interval', '5m')
        
        service = get_monitoring_service()
        history = service.get_metrics_history(start_time, end_time, interval)
        
        return jsonify({
            'success': True,
            'data': history
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@monitoring_bp.route('/alerts', methods=['GET'])
def get_active_alerts():
    """获取活跃告警"""
    try:
        service = get_monitoring_service()
        alerts = service.get_active_alerts()
        
        # 转换为字典格式
        alerts_data = []
        for alert in alerts:
            alerts_data.append({
                'alert_id': alert.alert_id,
                'rule_id': alert.rule_id,
                'name': alert.name,
                'description': alert.description,
                'level': alert.level.value,
                'timestamp': alert.timestamp.isoformat(),
                'metric_value': alert.metric_value,
                'threshold': alert.threshold,
                'resolved': alert.resolved
            })
        
        return jsonify({
            'success': True,
            'data': alerts_data
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@monitoring_bp.route('/alerts/<alert_id>/resolve', methods=['POST'])
def resolve_alert(alert_id):
    """解决告警"""
    try:
        service = get_monitoring_service()
        # 在实际实现中，这里会调用服务来解决告警
        # 为简化起见，我们直接返回成功
        return jsonify({
            'success': True,
            'message': f'Alert {alert_id} resolved'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@monitoring_bp.route('/performance-analysis', methods=['POST'])
def perform_performance_analysis():
    """执行性能分析"""
    try:
        analysis_type = request.json.get('analysis_type', 'full')
        target_id = request.json.get('target_id')
        
        # 获取当前指标
        service = get_monitoring_service()
        current_metrics = service.get_current_metrics()
        
        # 执行分析
        analyzer = get_performance_analyzer()
        report = analyzer.analyze(
            current_metrics['system'],
            current_metrics['gpu'],
            AnalysisType(analysis_type)
        )
        
        # 转换报告为字典格式
        report_data = {
            'report_id': report.report_id,
            'timestamp': report.timestamp.isoformat(),
            'analysis_type': report.analysis_type.value,
            'target_id': report.target_id,
            'duration_seconds': report.duration_seconds,
            'summary': report.summary,
            'overall_score': report.overall_score,
            'bottlenecks': [],
            'recommendations': []
        }
        
        # 转换瓶颈信息
        for bottleneck in report.bottlenecks:
            report_data['bottlenecks'].append({
                'bottleneck_id': bottleneck.bottleneck_id,
                'type': bottleneck.type.value,
                'severity': bottleneck.severity.value,
                'description': bottleneck.description,
                'metrics': bottleneck.metrics,
                'impact_score': bottleneck.impact_score,
                'affected_components': bottleneck.affected_components,
                'suggested_actions': bottleneck.suggested_actions
            })
        
        # 转换建议信息
        for recommendation in report.recommendations:
            report_data['recommendations'].append({
                'recommendation_id': recommendation.recommendation_id,
                'timestamp': recommendation.timestamp.isoformat(),
                'title': recommendation.title,
                'description': recommendation.description,
                'category': recommendation.category,
                'priority': recommendation.priority,
                'estimated_impact': recommendation.estimated_impact,
                'confidence': recommendation.confidence,
                'implementation_effort': recommendation.implementation_effort,
                'estimated_time_hours': recommendation.estimated_time_hours,
                'related_metrics': recommendation.related_metrics,
                'prerequisites': recommendation.prerequisites
            })
        
        return jsonify({
            'success': True,
            'data': report_data
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@monitoring_bp.route('/recommendations', methods=['GET'])
def get_optimization_recommendations():
    """获取优化建议"""
    try:
        # 获取当前指标
        service = get_monitoring_service()
        current_metrics = service.get_current_metrics()
        
        # 生成优化建议
        optimizer = get_resource_optimizer()
        recommendations = optimizer.generate_recommendations(
            current_metrics['system'],
            current_metrics['gpu']
        )
        
        # 转换建议为字典格式
        recommendations_data = []
        for recommendation in recommendations:
            recommendations_data.append({
                'recommendation_id': recommendation.recommendation_id,
                'timestamp': recommendation.timestamp.isoformat(),
                'category': recommendation.category,
                'priority': recommendation.priority,
                'confidence': recommendation.confidence,
                'title': recommendation.title,
                'description': recommendation.description,
                'action': recommendation.action,
                'expected_improvement': recommendation.expected_improvement,
                'estimated_cost': recommendation.estimated_cost,
                'implementation_effort': recommendation.implementation_effort,
                'affected_resources': recommendation.affected_resources
            })
        
        return jsonify({
            'success': True,
            'data': recommendations_data
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@monitoring_bp.route('/system-overview', methods=['GET'])
def get_system_overview():
    """获取系统概览"""
    try:
        service = get_monitoring_service()
        overview = service.get_status()
        
        # 获取当前指标
        current_metrics = service.get_current_metrics()
        
        # 添加关键指标
        if current_metrics['system']:
            overview['cpu_utilization'] = current_metrics['system'].cpu_percent
            overview['memory_utilization'] = current_metrics['system'].memory_percent
            overview['disk_utilization'] = current_metrics['system'].disk_percent
        
        return jsonify({
            'success': True,
            'data': overview
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@monitoring_bp.route('/health', methods=['GET'])
def health_check():
    """健康检查"""
    try:
        service = get_monitoring_service()
        status = service.get_status()
        
        return jsonify({
            'success': True,
            'data': {
                'status': 'healthy' if status['status'] == 'running' else 'unhealthy',
                'details': status
            }
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


def init_monitoring_api(app):
    """初始化监控API"""
    app.register_blueprint(monitoring_bp)
    print("监控API已注册")


def cleanup_monitoring_api():
    """清理监控API"""
    # 在实际应用中，这里可以添加清理逻辑
    print("监控API已清理")
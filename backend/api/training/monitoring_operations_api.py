"""
监控运维API
提供性能监控和运维自动化的REST API接口

支持租户级别的数据隔离和数据库持久化存储
"""
from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import logging

from backend.services.monitoring_operations_service import (
    get_monitoring_operations_service,
    MonitoringMetricType,
    AlertSeverity,
    AlertRule,
    AutomationTask
)

# 创建蓝图
monitoring_operations_bp = Blueprint('monitoring_operations', __name__, url_prefix='/api/v1/training/monitoring')
logger = logging.getLogger(__name__)


def _get_tenant_id() -> Optional[str]:
    """获取当前租户ID"""
    # 优先从请求头获取
    tenant_id = request.headers.get('X-Tenant-ID')
    if tenant_id:
        return tenant_id
    
    # 从查询参数获取
    tenant_id = request.args.get('tenant_id')
    if tenant_id:
        return tenant_id
    
    # 尝试从JWT获取
    try:
        from flask_jwt_extended import get_jwt
        claims = get_jwt()
        return claims.get('tenant_id')
    except Exception:
        pass
    
    return None


def _get_service():
    """获取监控运维服务实例"""
    return get_monitoring_operations_service()


@monitoring_operations_bp.route('/deployments/<deployment_id>/metrics', methods=['GET'])
@jwt_required()
def get_performance_metrics(deployment_id: str):
    """
    获取性能指标
    
    Args:
        deployment_id: 部署ID
        
    Query Parameters:
        metric_types: 指标类型列表(逗号分隔)
        save_to_db: 是否保存到数据库 (default: true)
        
    Returns:
        性能指标数据
    """
    try:
        user_id = get_jwt_identity()
        tenant_id = _get_tenant_id()
        logger.info(f"User {user_id} requesting metrics for deployment {deployment_id}")
        
        # 获取查询参数
        metric_types_str = request.args.get('metric_types')
        metric_types = None
        if metric_types_str:
            metric_types = metric_types_str.split(',')
        
        save_to_db = request.args.get('save_to_db', 'true').lower() == 'true'
        
        # 收集性能指标
        service = _get_service()
        report = service.collect_performance_metrics(
            deployment_id=deployment_id,
            metric_types=metric_types,
            tenant_id=tenant_id,
            save_to_db=save_to_db
        )
        
        return jsonify({
            'success': True,
            'data': {
                'deployment_id': report.deployment_id,
                'metrics': [
                    {
                        'type': metric.metric_type.value,
                        'value': metric.value,
                        'timestamp': metric.timestamp.isoformat(),
                        'metadata': metric.metadata
                    } for metric in report.metrics
                ],
                'time_range': {
                    'start': report.time_range['start'].isoformat(),
                    'end': report.time_range['end'].isoformat()
                },
                'summary': report.summary,
                'recommendations': report.recommendations
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Failed to get performance metrics: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Failed to get performance metrics: {str(e)}'
        }), 500


@monitoring_operations_bp.route('/system', methods=['GET'])
@jwt_required()
def get_system_monitoring():
    """
    获取系统监控信息
    
    Returns:
        系统监控数据
    """
    try:
        # 获取当前用户ID
        user_id = get_jwt_identity()
        logger.info(f"用户 {user_id} 请求获取系统监控信息")
        
        # 模拟系统监控数据
        system_data = {
            'cpu_usage': 45.2,
            'memory_usage': 68.5,
            'disk_usage': 32.1,
            'network_io': {
                'bytes_sent': 1024000,
                'bytes_received': 2048000
            },
            'uptime': '5 days, 12:30:45',
            'load_average': [1.2, 1.5, 1.8],
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify({
            'success': True,
            'data': system_data
        }), 200
        
    except Exception as e:
        logger.error(f"获取系统监控信息失败: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'获取系统监控信息失败: {str(e)}'
        }), 500


@monitoring_operations_bp.route('/training', methods=['GET'])
@jwt_required()
def get_training_monitoring():
    """
    获取训练监控信息
    
    Returns:
        训练监控数据
    """
    try:
        # 获取当前用户ID
        user_id = get_jwt_identity()
        logger.info(f"用户 {user_id} 请求获取训练监控信息")
        
        # 模拟训练监控数据
        training_data = {
            'active_jobs': 3,
            'completed_jobs': 15,
            'failed_jobs': 2,
            'queue_length': 5,
            'average_training_time': '2h 30m',
            'gpu_utilization': 78.5,
            'current_jobs': [
                {
                    'job_id': 'job_001',
                    'status': 'running',
                    'progress': 65.2,
                    'eta': '45 minutes'
                },
                {
                    'job_id': 'job_002',
                    'status': 'running',
                    'progress': 23.8,
                    'eta': '2 hours'
                }
            ],
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify({
            'success': True,
            'data': training_data
        }), 200
        
    except Exception as e:
        logger.error(f"获取训练监控信息失败: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'获取训练监控信息失败: {str(e)}'
        }), 500


@monitoring_operations_bp.route('/resources', methods=['GET'])
@jwt_required()
def get_resources_monitoring():
    """
    获取资源使用监控信息
    
    Returns:
        资源监控数据
    """
    try:
        # 获取当前用户ID
        user_id = get_jwt_identity()
        logger.info(f"用户 {user_id} 请求获取资源监控信息")
        
        # 模拟资源监控数据
        resources_data = {
            'gpu': {
                'total': 4,
                'available': 2,
                'utilization': [85.2, 92.1, 0.0, 0.0],
                'memory_usage': [7.2, 8.1, 0.0, 0.0],
                'temperature': [72, 78, 45, 46]
            },
            'cpu': {
                'cores': 16,
                'usage_per_core': [45.2, 38.7, 52.1, 41.8, 35.9, 48.3, 42.7, 39.1, 
                                  44.5, 37.2, 49.8, 43.6, 40.3, 46.9, 38.4, 41.7],
                'average_usage': 43.2
            },
            'memory': {
                'total_gb': 64,
                'used_gb': 43.8,
                'available_gb': 20.2,
                'usage_percentage': 68.4
            },
            'storage': {
                'total_tb': 2.0,
                'used_tb': 0.8,
                'available_tb': 1.2,
                'usage_percentage': 40.0
            },
            'network': {
                'bandwidth_mbps': 1000,
                'current_usage_mbps': 125.6,
                'peak_usage_mbps': 456.2
            },
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify({
            'success': True,
            'data': resources_data
        }), 200
        
    except Exception as e:
        logger.error(f"获取资源监控信息失败: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'获取资源监控信息失败: {str(e)}'
        }), 500


@monitoring_operations_bp.route('/alerts/rules', methods=['POST'])
@jwt_required()
def create_alert_rule():
    """
    创建告警规则
    
    Request Body:
        name: 规则名称
        metric_type: 指标类型
        threshold: 阈值
        operator: 操作符
        severity: 严重程度
        duration: 持续时间(秒)
        enabled: 是否启用
        description: 描述
        deployment_id: 绑定的部署ID (可选)
        
    Returns:
        操作结果
    """
    try:
        user_id = get_jwt_identity()
        tenant_id = _get_tenant_id()
        logger.info(f"User {user_id} creating alert rule")
        
        data = request.get_json()
        
        # 验证指标类型
        try:
            metric_type = MonitoringMetricType(data['metric_type'])
        except (KeyError, ValueError):
            return jsonify({
                'success': False,
                'error': f'Unsupported metric type: {data.get("metric_type")}'
            }), 400
        
        # 验证严重程度
        try:
            severity = AlertSeverity(data['severity'])
        except (KeyError, ValueError):
            return jsonify({
                'success': False,
                'error': f'Unsupported severity: {data.get("severity")}'
            }), 400
        
        # 创建告警规则
        rule = AlertRule(
            name=data['name'],
            metric_type=metric_type,
            threshold=float(data['threshold']),
            operator=data['operator'],
            severity=severity,
            duration=int(data.get('duration', 0)),
            enabled=data.get('enabled', True),
            description=data.get('description', '')
        )
        
        service = _get_service()
        result = service.create_alert_rule(
            rule=rule,
            tenant_id=tenant_id,
            user_id=user_id,
            deployment_id=data.get('deployment_id')
        )
        
        if result.get('success'):
            return jsonify({
                'success': True,
                'message': f'Alert rule {rule.name} created successfully',
                'data': result
            }), 201
        else:
            return jsonify({
                'success': False,
                'error': result.get('error', 'Failed to create alert rule')
            }), 500
        
    except Exception as e:
        logger.error(f"Failed to create alert rule: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Failed to create alert rule: {str(e)}'
        }), 500


@monitoring_operations_bp.route('/alerts/rules', methods=['GET'])
@jwt_required()
def get_alert_rules():
    """
    获取告警规则列表
    
    Query Parameters:
        enabled_only: 是否只返回启用的规则
        deployment_id: 部署ID过滤
        limit: 返回数量限制
        offset: 偏移量
        
    Returns:
        告警规则列表
    """
    try:
        user_id = get_jwt_identity()
        tenant_id = _get_tenant_id()
        
        enabled_only = request.args.get('enabled_only', 'false').lower() == 'true'
        deployment_id = request.args.get('deployment_id')
        limit = request.args.get('limit', 100, type=int)
        offset = request.args.get('offset', 0, type=int)
        
        service = _get_service()
        result = service.get_alert_rules_list(
            tenant_id=tenant_id,
            enabled_only=enabled_only,
            deployment_id=deployment_id,
            limit=limit,
            offset=offset
        )
        
        return jsonify({
            'success': True,
            'data': {
                'rules': result.get('rules', []),
                'total': result.get('total', 0),
                'limit': limit,
                'offset': offset
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Failed to get alert rules: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Failed to get alert rules: {str(e)}'
        }), 500


@monitoring_operations_bp.route('/alerts/rules/<rule_id>', methods=['GET'])
@jwt_required()
def get_alert_rule(rule_id: str):
    """获取指定告警规则"""
    try:
        tenant_id = _get_tenant_id()
        service = _get_service()
        
        rule = service.get_alert_rule_by_id(rule_id, tenant_id)
        
        if rule:
            return jsonify({
                'success': True,
                'data': rule
            }), 200
        else:
            return jsonify({
                'success': False,
                'error': 'Alert rule not found'
            }), 404
        
    except Exception as e:
        logger.error(f"Failed to get alert rule: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Failed to get alert rule: {str(e)}'
        }), 500


@monitoring_operations_bp.route('/alerts/rules/<rule_id>', methods=['PUT'])
@jwt_required()
def update_alert_rule(rule_id: str):
    """更新告警规则"""
    try:
        tenant_id = _get_tenant_id()
        data = request.get_json()
        
        service = _get_service()
        result = service.update_alert_rule(rule_id, tenant_id, data)
        
        if result:
            return jsonify({
                'success': True,
                'data': result,
                'message': 'Alert rule updated successfully'
            }), 200
        else:
            return jsonify({
                'success': False,
                'error': 'Alert rule not found or update failed'
            }), 404
        
    except Exception as e:
        logger.error(f"Failed to update alert rule: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Failed to update alert rule: {str(e)}'
        }), 500


@monitoring_operations_bp.route('/alerts/rules/<rule_id>', methods=['DELETE'])
@jwt_required()
def delete_alert_rule(rule_id: str):
    """删除告警规则"""
    try:
        tenant_id = _get_tenant_id()
        service = _get_service()
        
        success = service.delete_alert_rule(rule_id, tenant_id)
        
        if success:
            return jsonify({
                'success': True,
                'message': 'Alert rule deleted successfully'
            }), 200
        else:
            return jsonify({
                'success': False,
                'error': 'Alert rule not found'
            }), 404
        
    except Exception as e:
        logger.error(f"Failed to delete alert rule: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Failed to delete alert rule: {str(e)}'
        }), 500


@monitoring_operations_bp.route('/deployments/<deployment_id>/alerts', methods=['GET'])
@jwt_required()
def check_alerts(deployment_id: str):
    """
    检查并获取告警
    
    Args:
        deployment_id: 部署ID
        
    Query Parameters:
        check_now: 是否立即检查告警规则 (default: true)
        
    Returns:
        告警列表
    """
    try:
        user_id = get_jwt_identity()
        tenant_id = _get_tenant_id()
        logger.info(f"User {user_id} checking alerts for deployment {deployment_id}")
        
        check_now = request.args.get('check_now', 'true').lower() == 'true'
        
        service = _get_service()
        
        if check_now:
            # 执行实时告警检查
            alerts = service.check_alerts(deployment_id, tenant_id)
            
            return jsonify({
                'success': True,
                'data': {
                    'deployment_id': deployment_id,
                    'alerts': [
                        {
                            'alert_id': alert.alert_id,
                            'rule_name': alert.rule_name,
                            'severity': alert.severity.value,
                            'message': alert.message,
                            'timestamp': alert.timestamp.isoformat(),
                            'metric_value': alert.metric_value,
                            'resolved': alert.resolved
                        } for alert in alerts
                    ],
                    'count': len(alerts)
                }
            }), 200
        else:
            # 从数据库获取告警历史
            result = service.get_alert_history_list(
                tenant_id=tenant_id,
                deployment_id=deployment_id,
                resolved=False,
                limit=50
            )
            
            return jsonify({
                'success': True,
                'data': {
                    'deployment_id': deployment_id,
                    'alerts': result.get('alerts', []),
                    'count': result.get('total', 0)
                }
            }), 200
        
    except Exception as e:
        logger.error(f"Failed to check alerts: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Failed to check alerts: {str(e)}'
        }), 500


@monitoring_operations_bp.route('/deployments/<deployment_id>/alerts/<alert_id>/resolve', methods=['POST'])
@jwt_required()
def resolve_alert(deployment_id: str, alert_id: str):
    """解决告警"""
    try:
        user_id = get_jwt_identity()
        tenant_id = _get_tenant_id()
        
        data = request.get_json() or {}
        resolution_notes = data.get('resolution_notes')
        
        service = _get_service()
        result = service.resolve_alert_by_id(alert_id, tenant_id, resolution_notes)
        
        if result:
            return jsonify({
                'success': True,
                'data': result,
                'message': 'Alert resolved successfully'
            }), 200
        else:
            return jsonify({
                'success': False,
                'error': 'Alert not found'
            }), 404
        
    except Exception as e:
        logger.error(f"Failed to resolve alert: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Failed to resolve alert: {str(e)}'
        }), 500


@monitoring_operations_bp.route('/deployments/<deployment_id>/alerts/<alert_id>/acknowledge', methods=['POST'])
@jwt_required()
def acknowledge_alert(deployment_id: str, alert_id: str):
    """确认告警"""
    try:
        user_id = get_jwt_identity()
        tenant_id = _get_tenant_id()
        
        service = _get_service()
        result = service.acknowledge_alert(alert_id, tenant_id, user_id)
        
        if result:
            return jsonify({
                'success': True,
                'data': result,
                'message': 'Alert acknowledged successfully'
            }), 200
        else:
            return jsonify({
                'success': False,
                'error': 'Alert not found'
            }), 404
        
    except Exception as e:
        logger.error(f"Failed to acknowledge alert: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Failed to acknowledge alert: {str(e)}'
        }), 500


@monitoring_operations_bp.route('/alerts/statistics', methods=['GET'])
@jwt_required()
def get_alert_statistics():
    """获取告警统计信息"""
    try:
        tenant_id = _get_tenant_id()
        deployment_id = request.args.get('deployment_id')
        
        # 解析时间范围
        start_time_str = request.args.get('start_time')
        end_time_str = request.args.get('end_time')
        
        start_time = None
        end_time = None
        if start_time_str:
            start_time = datetime.fromisoformat(start_time_str.replace('Z', '+00:00'))
        if end_time_str:
            end_time = datetime.fromisoformat(end_time_str.replace('Z', '+00:00'))
        
        service = _get_service()
        stats = service.get_alert_statistics(
            tenant_id=tenant_id,
            deployment_id=deployment_id,
            start_time=start_time,
            end_time=end_time
        )
        
        return jsonify({
            'success': True,
            'data': stats
        }), 200
        
    except Exception as e:
        logger.error(f"Failed to get alert statistics: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Failed to get alert statistics: {str(e)}'
        }), 500


@monitoring_operations_bp.route('/deployments/<deployment_id>/automation', methods=['POST'])
@jwt_required()
def execute_automation_task(deployment_id: str):
    """
    执行自动化任务
    
    Args:
        deployment_id: 部署ID
        
    Request Body:
        task_type: 任务类型 (auto_scaling, fault_recovery, capacity_planning, resource_optimization, alert_management)
        parameters: 参数
        alert_id: 触发告警ID (可选)
        
    Returns:
        任务执行结果
    """
    try:
        user_id = get_jwt_identity()
        tenant_id = _get_tenant_id()
        logger.info(f"User {user_id} executing automation task for deployment {deployment_id}")
        
        data = request.get_json()
        task_type = data.get('task_type')
        parameters = data.get('parameters', {})
        alert_id = data.get('alert_id')
        
        if not task_type:
            return jsonify({
                'success': False,
                'error': 'Missing task_type'
            }), 400
        
        # 验证任务类型
        valid_types = ['auto_scaling', 'fault_recovery', 'capacity_planning', 'resource_optimization', 'alert_management']
        if task_type not in valid_types:
            return jsonify({
                'success': False,
                'error': f'Invalid task_type. Must be one of: {valid_types}'
            }), 400
        
        service = _get_service()
        task = service.execute_automation_task(
            deployment_id=deployment_id,
            task_type=task_type,
            parameters=parameters,
            tenant_id=tenant_id,
            user_id=user_id,
            alert_id=alert_id
        )
        
        return jsonify({
            'success': True,
            'data': {
                'task_id': task.task_id,
                'name': task.name,
                'task_type': task.task_type,
                'status': task.status,
                'deployment_id': task.deployment_id,
                'created_at': task.created_at.isoformat(),
                'completed_at': task.completed_at.isoformat() if task.completed_at else None,
                'result': task.result
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Failed to execute automation task: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Failed to execute automation task: {str(e)}'
        }), 500


@monitoring_operations_bp.route('/automation/tasks', methods=['GET'])
@jwt_required()
def get_automation_tasks():
    """
    获取自动化任务列表
    
    Query Parameters:
        status: 状态过滤
        task_type: 任务类型过滤
        deployment_id: 部署ID过滤
        limit: 返回数量限制
        offset: 偏移量
        
    Returns:
        任务列表
    """
    try:
        tenant_id = _get_tenant_id()
        
        status = request.args.get('status')
        task_type = request.args.get('task_type')
        deployment_id = request.args.get('deployment_id')
        limit = request.args.get('limit', 50, type=int)
        offset = request.args.get('offset', 0, type=int)
        
        service = _get_service()
        result = service.get_task_list(
            tenant_id=tenant_id,
            status=status,
            task_type=task_type,
            deployment_id=deployment_id,
            limit=limit,
            offset=offset
        )
        
        return jsonify({
            'success': True,
            'data': {
                'tasks': result.get('tasks', []),
                'total': result.get('total', 0),
                'limit': limit,
                'offset': offset
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Failed to get automation tasks: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Failed to get automation tasks: {str(e)}'
        }), 500


@monitoring_operations_bp.route('/automation/tasks/statistics', methods=['GET'])
@jwt_required()
def get_task_statistics():
    """获取任务统计信息"""
    try:
        tenant_id = _get_tenant_id()
        deployment_id = request.args.get('deployment_id')
        
        service = _get_service()
        stats = service.get_task_statistics(tenant_id, deployment_id)
        
        return jsonify({
            'success': True,
            'data': stats
        }), 200
        
    except Exception as e:
        logger.error(f"Failed to get task statistics: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Failed to get task statistics: {str(e)}'
        }), 500


@monitoring_operations_bp.route('/deployments/<deployment_id>/analytics', methods=['GET'])
@jwt_required()
def get_deployment_analytics(deployment_id: str):
    """
    获取部署分析数据
    
    Args:
        deployment_id: 部署ID
        
    Query Parameters:
        start_time: 开始时间
        end_time: 结束时间
        save_report: 是否保存报告到数据库 (default: true)
        
    Returns:
        分析数据
    """
    try:
        user_id = get_jwt_identity()
        tenant_id = _get_tenant_id()
        logger.info(f"User {user_id} requesting analytics for deployment {deployment_id}")
        
        # 获取查询参数
        start_time_str = request.args.get('start_time')
        end_time_str = request.args.get('end_time')
        save_report = request.args.get('save_report', 'true').lower() == 'true'
        
        # 解析时间参数
        time_range = {}
        if start_time_str:
            time_range['start'] = datetime.fromisoformat(start_time_str.replace('Z', '+00:00'))
        else:
            time_range['start'] = datetime.utcnow() - timedelta(days=7)
            
        if end_time_str:
            time_range['end'] = datetime.fromisoformat(end_time_str.replace('Z', '+00:00'))
        else:
            time_range['end'] = datetime.utcnow()
        
        service = _get_service()
        analytics = service.get_deployment_analytics(
            deployment_id=deployment_id,
            time_range=time_range,
            tenant_id=tenant_id,
            save_report=save_report
        )
        
        return jsonify({
            'success': True,
            'data': analytics
        }), 200
        
    except Exception as e:
        logger.error(f"Failed to get analytics: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Failed to get analytics: {str(e)}'
        }), 500


@monitoring_operations_bp.route('/reports', methods=['GET'])
@jwt_required()
def get_monitoring_reports():
    """
    获取监控报告列表
    
    Query Parameters:
        report_type: 报告类型过滤
        deployment_id: 部署ID过滤
        limit: 返回数量限制
        offset: 偏移量
        
    Returns:
        报告列表
    """
    try:
        tenant_id = _get_tenant_id()
        
        report_type = request.args.get('report_type')
        deployment_id = request.args.get('deployment_id')
        limit = request.args.get('limit', 20, type=int)
        offset = request.args.get('offset', 0, type=int)
        
        service = _get_service()
        result = service.get_report_list(
            tenant_id=tenant_id,
            report_type=report_type,
            deployment_id=deployment_id,
            limit=limit,
            offset=offset
        )
        
        return jsonify({
            'success': True,
            'data': {
                'reports': result.get('reports', []),
                'total': result.get('total', 0),
                'limit': limit,
                'offset': offset
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Failed to get monitoring reports: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Failed to get monitoring reports: {str(e)}'
        }), 500


@monitoring_operations_bp.route('/reports/<report_id>', methods=['GET'])
@jwt_required()
def get_monitoring_report(report_id: str):
    """获取指定监控报告"""
    try:
        tenant_id = _get_tenant_id()
        service = _get_service()
        
        report = service.get_report_by_id(report_id, tenant_id)
        
        if report:
            return jsonify({
                'success': True,
                'data': report
            }), 200
        else:
            return jsonify({
                'success': False,
                'error': 'Report not found'
            }), 404
        
    except Exception as e:
        logger.error(f"Failed to get monitoring report: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Failed to get monitoring report: {str(e)}'
        }), 500


@monitoring_operations_bp.route('/deployments/<deployment_id>/metrics/history', methods=['GET'])
@jwt_required()
def get_metrics_history(deployment_id: str):
    """
    获取历史指标数据
    
    Args:
        deployment_id: 部署ID
        
    Query Parameters:
        metric_types: 指标类型列表(逗号分隔)
        start_time: 开始时间
        end_time: 结束时间
        limit: 返回数量限制
        offset: 偏移量
        
    Returns:
        历史指标数据
    """
    try:
        tenant_id = _get_tenant_id()
        
        metric_types_str = request.args.get('metric_types')
        metric_types = metric_types_str.split(',') if metric_types_str else None
        
        start_time_str = request.args.get('start_time')
        end_time_str = request.args.get('end_time')
        limit = request.args.get('limit', 100, type=int)
        offset = request.args.get('offset', 0, type=int)
        
        start_time = None
        end_time = None
        if start_time_str:
            start_time = datetime.fromisoformat(start_time_str.replace('Z', '+00:00'))
        if end_time_str:
            end_time = datetime.fromisoformat(end_time_str.replace('Z', '+00:00'))
        
        service = _get_service()
        result = service.get_metrics_history(
            tenant_id=tenant_id,
            deployment_id=deployment_id,
            metric_types=metric_types,
            start_time=start_time,
            end_time=end_time,
            limit=limit,
            offset=offset
        )
        
        return jsonify({
            'success': True,
            'data': {
                'deployment_id': deployment_id,
                'metrics': result.get('metrics', []),
                'total': result.get('total', 0),
                'limit': limit,
                'offset': offset
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Failed to get metrics history: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Failed to get metrics history: {str(e)}'
        }), 500


@monitoring_operations_bp.route('/deployments/<deployment_id>/metrics/latest', methods=['GET'])
@jwt_required()
def get_latest_metrics(deployment_id: str):
    """获取最新指标值"""
    try:
        tenant_id = _get_tenant_id()
        
        metric_types_str = request.args.get('metric_types')
        metric_types = metric_types_str.split(',') if metric_types_str else None
        
        service = _get_service()
        latest = service.get_latest_metrics(tenant_id, deployment_id, metric_types)
        
        return jsonify({
            'success': True,
            'data': {
                'deployment_id': deployment_id,
                'metrics': latest
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Failed to get latest metrics: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Failed to get latest metrics: {str(e)}'
        }), 500


@monitoring_operations_bp.route('/metrics/statistics', methods=['GET'])
@jwt_required()
def get_metrics_statistics():
    """获取指标统计信息"""
    try:
        tenant_id = _get_tenant_id()
        deployment_id = request.args.get('deployment_id')
        
        start_time_str = request.args.get('start_time')
        end_time_str = request.args.get('end_time')
        
        start_time = None
        end_time = None
        if start_time_str:
            start_time = datetime.fromisoformat(start_time_str.replace('Z', '+00:00'))
        if end_time_str:
            end_time = datetime.fromisoformat(end_time_str.replace('Z', '+00:00'))
        
        service = _get_service()
        stats = service.get_metrics_statistics(
            tenant_id=tenant_id,
            deployment_id=deployment_id,
            start_time=start_time,
            end_time=end_time
        )
        
        return jsonify({
            'success': True,
            'data': stats
        }), 200
        
    except Exception as e:
        logger.error(f"Failed to get metrics statistics: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Failed to get metrics statistics: {str(e)}'
        }), 500


@monitoring_operations_bp.route('/cleanup', methods=['POST'])
@jwt_required()
def cleanup_old_data():
    """清理过期的监控数据"""
    try:
        user_id = get_jwt_identity()
        tenant_id = _get_tenant_id()
        
        data = request.get_json() or {}
        retention_days = data.get('retention_days', 30)
        
        service = _get_service()
        result = service.cleanup_old_data(tenant_id, retention_days)
        
        logger.info(f"User {user_id} cleaned up old monitoring data: {result}")
        
        return jsonify({
            'success': True,
            'data': {
                'cleaned': result,
                'retention_days': retention_days
            },
            'message': 'Cleanup completed successfully'
        }), 200
        
    except Exception as e:
        logger.error(f"Failed to cleanup old data: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Failed to cleanup old data: {str(e)}'
        }), 500


@monitoring_operations_bp.route('/metrics/types', methods=['GET'])
@jwt_required()
def get_metric_types():
    """
    获取支持的监控指标类型
    
    Returns:
        指标类型列表
    """
    try:
        metric_types = [metric.value for metric in MonitoringMetricType]
        severities = [severity.value for severity in AlertSeverity]
        
        return jsonify({
            'success': True,
            'data': {
                'metric_types': metric_types,
                'alert_severities': severities
            }
        }), 200
        
    except Exception as e:
        logger.error(f"获取指标类型失败: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'获取指标类型失败: {str(e)}'
        }), 500


@monitoring_operations_bp.route('/deployments/<deployment_id>/alerts/history', methods=['GET'])
@jwt_required()
def get_alerts_history(deployment_id: str):
    """
    获取告警历史
    
    Args:
        deployment_id: 部署ID
        
    Query Parameters:
        limit: 限制返回记录数
        severity: 严重程度过滤
        
    Returns:
        告警历史记录
    """
    try:
        # 获取当前用户ID
        user_id = get_jwt_identity()
        logger.info(f"用户 {user_id} 请求获取部署 {deployment_id} 的告警历史")
        
        # 获取查询参数
        limit = request.args.get('limit', 50, type=int)
        severity_str = request.args.get('severity')
        
        # 调用服务获取告警历史
        try:
            service = _get_service()
            alerts_history = service.get_alerts_history(
                deployment_id=deployment_id,
                limit=limit,
                severity=AlertSeverity(severity_str) if severity_str else None
            )
            
            alerts = []
            for alert in alerts_history:
                alerts.append({
                    'alert_id': alert.alert_id,
                    'deployment_id': alert.deployment_id,
                    'severity': alert.severity.value,
                    'message': alert.message,
                    'timestamp': alert.timestamp.isoformat(),
                    'resolved': alert.resolved
                })
        except Exception as e:
            logger.warning(f"获取告警历史失败，返回空列表: {e}")
            alerts = []
        
        return jsonify({
            'success': True,
            'data': {
                'alerts': alerts,
                'count': len(alerts)
            }
        }), 200
        
    except Exception as e:
        logger.error(f"获取告警历史失败: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'获取告警历史失败: {str(e)}'
        }), 500


@monitoring_operations_bp.route('/automation/tasks/<task_id>', methods=['GET'])
@jwt_required()
def get_automation_task_status(task_id: str):
    """
    获取自动化任务状态
    
    Args:
        task_id: 任务ID
        
    Returns:
        任务状态信息
    """
    try:
        user_id = get_jwt_identity()
        tenant_id = _get_tenant_id()
        logger.info(f"User {user_id} requesting task status for {task_id}")
        
        service = _get_service()
        task = service.get_task_by_id(task_id, tenant_id)
        
        if not task:
            return jsonify({
                'success': False,
                'error': f'Task {task_id} not found'
            }), 404
        
        return jsonify({
            'success': True,
            'data': task
        }), 200
        
    except Exception as e:
        logger.error(f"Failed to get task status: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Failed to get task status: {str(e)}'
        }), 500
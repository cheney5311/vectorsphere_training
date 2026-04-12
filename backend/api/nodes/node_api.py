#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""节点管理 API

提供完整的节点管理REST API接口：
- 节点注册/注销
- 心跳处理
- 节点CRUD操作
- 健康检查
- 集群状态
- 事件管理
"""

import sys
import os
import logging
from datetime import datetime
from functools import wraps

from flask import Blueprint, request, jsonify, g
from flask_jwt_extended import jwt_required, get_jwt_identity

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.utils.response import success_response, error_response
from backend.services.node_service import get_node_service, NodeService

logger = logging.getLogger(__name__)

# Blueprint定义
node_bp = Blueprint('nodes', __name__, url_prefix='/api/v1/nodes')


# ==================== 辅助装饰器 ====================

def handle_api_errors(f):
    """API错误处理装饰器"""
    @wraps(f)
    def wrapper(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except ValueError as e:
            logger.warning(f"Validation error in {f.__name__}: {e}")
            return error_response(str(e), 400)
        except Exception as e:
            logger.error(f"Error in {f.__name__}: {e}", exc_info=True)
            return error_response(f"Internal server error: {str(e)}", 500)
    return wrapper


def get_tenant_id():
    """获取当前租户ID"""
    return request.headers.get('X-Tenant-ID') or getattr(g, 'tenant_id', None)


def get_service() -> NodeService:
    """获取节点服务实例"""
    return get_node_service(use_memory=True)


# ==================== 心跳接口 ====================

@node_bp.route('/heartbeat', methods=['POST'])
@jwt_required()
@handle_api_errors
def node_heartbeat():
    """节点心跳接口
    
    Agent定期调用，上报节点状态和指标
    
    Request Body:
        node_id: 节点ID (必需)
        hostname: 主机名 (必需)
        ip_address: IP地址
        port: 端口
        cpu_count: CPU核心数
        memory_total: 总内存(MB)
        memory_used: 已用内存(MB)
        disk_total: 总磁盘(MB)
        disk_used: 已用磁盘(MB)
        cpu_utilization: CPU利用率(0-100)
        memory_utilization: 内存利用率(0-100)
        disk_utilization: 磁盘利用率(0-100)
        gpus: GPU信息列表
        running_tasks: 运行中的任务数
        labels: 标签字典
        annotations: 注解字典
        metrics: 额外指标
    
    Returns:
        心跳处理结果
    """
    data = request.get_json() or {}
    
    node_id = data.get('node_id')
    hostname = data.get('hostname')
    
    if not node_id:
        return error_response('node_id is required', 400)
    if not hostname:
        return error_response('hostname is required', 400)
    
    service = get_service()
    result = service.process_heartbeat(
        node_id=node_id,
        hostname=hostname,
        ip_address=data.get('ip_address'),
        port=data.get('port', 22),
        cpu_count=data.get('cpu_count'),
        memory_total_mb=data.get('memory_total'),
        disk_total_mb=data.get('disk_total'),
        cpu_utilization=data.get('cpu_utilization'),
        memory_used_mb=data.get('memory_used'),
        memory_utilization=data.get('memory_utilization'),
        disk_used_mb=data.get('disk_used'),
        disk_utilization=data.get('disk_utilization'),
        gpu_info=data.get('gpus', []),
        running_tasks=data.get('running_tasks'),
        labels=data.get('labels', {}),
        annotations=data.get('annotations', {}),
        metrics=data.get('metrics'),
        tenant_id=get_tenant_id()
    )
    
    if result['success']:
        return success_response(
            data={
                'latency_ms': result.get('latency_ms'),
                'node_status': result.get('node_status')
            },
            message=result['message']
        )
    else:
        return error_response(result['message'], 400)


# ==================== 节点CRUD接口 ====================

@node_bp.route('', methods=['GET'])
@jwt_required()
@handle_api_errors
def list_nodes():
    """列出所有节点
    
    Query Parameters:
        status: 状态过滤 (healthy/unhealthy/offline/maintenance)
        node_type: 类型过滤 (master/worker/gpu/storage/edge)
        node_role: 角色过滤 (training/inference/data/general)
        label_key: 标签键
        label_value: 标签值
        limit: 返回数量 (默认100)
        offset: 偏移量 (默认0)
    
    Returns:
        节点列表
    """
    status = request.args.get('status')
    node_type = request.args.get('node_type')
    node_role = request.args.get('node_role')
    label_key = request.args.get('label_key')
    label_value = request.args.get('label_value')
    limit = request.args.get('limit', 100, type=int)
    offset = request.args.get('offset', 0, type=int)
    
    labels = None
    if label_key and label_value:
        labels = {label_key: label_value}
    
    service = get_service()
    result = service.list_nodes(
        tenant_id=get_tenant_id(),
        status=status,
        node_type=node_type,
        node_role=node_role,
        labels=labels,
        limit=limit,
        offset=offset
    )
    
    return success_response(data=result)


@node_bp.route('/<node_id>', methods=['GET'])
@jwt_required()
@handle_api_errors
def get_node(node_id: str):
    """获取节点详情
    
    Path Parameters:
        node_id: 节点ID
    
    Returns:
        节点详细信息，包括GPU、心跳和事件
    """
    service = get_service()
    node = service.get_node(node_id)
    
    if not node:
        return error_response(f'Node not found: {node_id}', 404)
    
    return success_response(data=node)


@node_bp.route('', methods=['POST'])
@jwt_required()
@handle_api_errors
def register_node():
    """注册新节点
    
    Request Body:
        node_id: 节点ID (必需)
        hostname: 主机名 (必需)
        ip_address: IP地址 (必需)
        port: 端口 (默认22)
        node_type: 节点类型 (worker/master/gpu/storage/edge)
        node_role: 节点角色 (training/inference/data/general)
        cpu_count: CPU核心数
        memory_total_mb: 总内存(MB)
        disk_total_mb: 总磁盘(MB)
        gpu_info: GPU信息列表
        labels: 标签字典
        annotations: 注解字典
        max_tasks: 最大任务数
    
    Returns:
        注册结果
    """
    data = request.get_json() or {}
    
    node_id = data.get('node_id')
    hostname = data.get('hostname')
    ip_address = data.get('ip_address')
    
    if not node_id:
        return error_response('node_id is required', 400)
    if not hostname:
        return error_response('hostname is required', 400)
    if not ip_address:
        return error_response('ip_address is required', 400)
    
    service = get_service()
    result = service.register_node(
        node_id=node_id,
        hostname=hostname,
        ip_address=ip_address,
        port=data.get('port', 22),
        node_type=data.get('node_type', 'worker'),
        node_role=data.get('node_role', 'general'),
        cpu_count=data.get('cpu_count', 0),
        memory_total_mb=data.get('memory_total_mb', 0),
        disk_total_mb=data.get('disk_total_mb', 0),
        gpu_info=data.get('gpu_info', []),
        labels=data.get('labels', {}),
        annotations=data.get('annotations', {}),
        tenant_id=get_tenant_id(),
        max_tasks=data.get('max_tasks', 1)
    )
    
    if result['success']:
        return success_response(data=result['node'], message=result['message']), 201
    else:
        return error_response(result['message'], 400)


@node_bp.route('/<node_id>', methods=['PUT'])
@jwt_required()
@handle_api_errors
def update_node(node_id: str):
    """更新节点信息
    
    Path Parameters:
        node_id: 节点ID
    
    Request Body:
        hostname: 主机名
        ip_address: IP地址
        port: 端口
        node_type: 节点类型
        node_role: 节点角色
        cpu_count: CPU核心数
        memory_total_mb: 总内存(MB)
        disk_total_mb: 总磁盘(MB)
        gpu_info: GPU信息列表
        labels: 标签字典
        annotations: 注解字典
        is_schedulable: 是否可调度
        max_tasks: 最大任务数
    
    Returns:
        更新结果
    """
    data = request.get_json() or {}
    
    service = get_service()
    result = service.update_node(node_id, **data)
    
    if result['success']:
        return success_response(data=result.get('node'), message=result['message'])
    else:
        return error_response(result['message'], 400)


@node_bp.route('/<node_id>', methods=['DELETE'])
@jwt_required()
@handle_api_errors
def unregister_node(node_id: str):
    """注销节点
    
    Path Parameters:
        node_id: 节点ID
    
    Query Parameters:
        reason: 注销原因
    
    Returns:
        注销结果
    """
    reason = request.args.get('reason', 'manual')
    
    service = get_service()
    result = service.unregister_node(node_id, reason=reason)
    
    if result['success']:
        return success_response(message=result['message'])
    else:
        return error_response(result['message'], 400)


# ==================== 节点状态管理接口 ====================

@node_bp.route('/<node_id>/cordon', methods=['POST'])
@jwt_required()
@handle_api_errors
def cordon_node(node_id: str):
    """隔离节点
    
    阻止新任务调度到此节点
    
    Path Parameters:
        node_id: 节点ID
    
    Request Body:
        reason: 隔离原因
    
    Returns:
        操作结果
    """
    data = request.get_json() or {}
    reason = data.get('reason', 'manual')
    
    service = get_service()
    result = service.cordon_node(node_id, reason=reason)
    
    if result['success']:
        return success_response(message=result['message'])
    else:
        return error_response(result['message'], 400)


@node_bp.route('/<node_id>/uncordon', methods=['POST'])
@jwt_required()
@handle_api_errors
def uncordon_node(node_id: str):
    """解除节点隔离
    
    允许新任务调度到此节点
    
    Path Parameters:
        node_id: 节点ID
    
    Returns:
        操作结果
    """
    service = get_service()
    result = service.uncordon_node(node_id)
    
    if result['success']:
        return success_response(message=result['message'])
    else:
        return error_response(result['message'], 400)


@node_bp.route('/<node_id>/drain', methods=['POST'])
@jwt_required()
@handle_api_errors
def drain_node(node_id: str):
    """排空节点
    
    迁移运行中的任务并阻止新任务调度
    
    Path Parameters:
        node_id: 节点ID
    
    Request Body:
        force: 是否强制排空 (默认false)
    
    Returns:
        操作结果
    """
    data = request.get_json() or {}
    force = data.get('force', False)
    
    service = get_service()
    result = service.drain_node(node_id, force=force)
    
    if result['success']:
        return success_response(
            data={'running_tasks': result.get('running_tasks', 0)},
            message=result['message']
        )
    else:
        return error_response(result['message'], 400)


# ==================== 健康检查接口 ====================

@node_bp.route('/<node_id>/health', methods=['GET'])
@jwt_required()
@handle_api_errors
def check_node_health(node_id: str):
    """检查节点健康状态
    
    Path Parameters:
        node_id: 节点ID
    
    Returns:
        健康检查结果
    """
    service = get_service()
    result = service.check_node_health(node_id)
    
    return success_response(data=result)


@node_bp.route('/health/check-all', methods=['POST'])
@jwt_required()
@handle_api_errors
def run_health_checks():
    """运行所有节点的健康检查
    
    Returns:
        健康检查摘要
    """
    service = get_service()
    result = service.run_health_checks()
    
    return success_response(data=result)


# ==================== 集群状态接口 ====================

@node_bp.route('/cluster/summary', methods=['GET'])
@jwt_required()
@handle_api_errors
def get_cluster_summary():
    """获取集群摘要
    
    Returns:
        集群状态摘要，包括节点、GPU、事件统计
    """
    service = get_service()
    summary = service.get_cluster_summary(tenant_id=get_tenant_id())
    
    return success_response(data=summary)


@node_bp.route('/cluster/schedulable', methods=['GET'])
@jwt_required()
@handle_api_errors
def get_schedulable_nodes():
    """获取可调度节点
    
    Query Parameters:
        required_gpus: 需要的GPU数量
        required_memory_mb: 需要的内存(MB)
        label_key: 标签键
        label_value: 标签值
    
    Returns:
        可调度节点列表
    """
    required_gpus = request.args.get('required_gpus', 0, type=int)
    required_memory_mb = request.args.get('required_memory_mb', 0, type=int)
    label_key = request.args.get('label_key')
    label_value = request.args.get('label_value')
    
    labels = None
    if label_key and label_value:
        labels = {label_key: label_value}
    
    service = get_service()
    nodes = service.get_schedulable_nodes(
        tenant_id=get_tenant_id(),
        required_gpus=required_gpus,
        required_memory_mb=required_memory_mb,
        labels=labels
    )
    
    return success_response(data={'nodes': nodes, 'count': len(nodes)})


# ==================== 心跳历史接口 ====================

@node_bp.route('/<node_id>/heartbeats', methods=['GET'])
@jwt_required()
@handle_api_errors
def get_heartbeat_history(node_id: str):
    """获取节点心跳历史
    
    Path Parameters:
        node_id: 节点ID
    
    Query Parameters:
        start_time: 开始时间 (ISO格式)
        end_time: 结束时间 (ISO格式)
        limit: 返回数量 (默认100)
    
    Returns:
        心跳历史列表
    """
    start_time = request.args.get('start_time')
    end_time = request.args.get('end_time')
    limit = request.args.get('limit', 100, type=int)
    
    if start_time:
        start_time = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
    if end_time:
        end_time = datetime.fromisoformat(end_time.replace('Z', '+00:00'))
    
    service = get_service()
    heartbeats = service.get_heartbeat_history(
        node_id=node_id,
        start_time=start_time,
        end_time=end_time,
        limit=limit
    )
    
    return success_response(data={'heartbeats': heartbeats, 'count': len(heartbeats)})


# ==================== 事件接口 ====================

@node_bp.route('/events', methods=['GET'])
@jwt_required()
@handle_api_errors
def get_events():
    """获取节点事件列表
    
    Query Parameters:
        node_id: 节点ID (可选)
        event_type: 事件类型
        severity: 严重级别 (info/warning/error/critical)
        limit: 返回数量 (默认100)
        offset: 偏移量 (默认0)
    
    Returns:
        事件列表
    """
    node_id = request.args.get('node_id')
    event_type = request.args.get('event_type')
    severity = request.args.get('severity')
    limit = request.args.get('limit', 100, type=int)
    offset = request.args.get('offset', 0, type=int)
    
    service = get_service()
    result = service.get_node_events(
        node_id=node_id,
        event_type=event_type,
        severity=severity,
        limit=limit,
        offset=offset
    )
    
    return success_response(data=result)


@node_bp.route('/<node_id>/events', methods=['GET'])
@jwt_required()
@handle_api_errors
def get_node_events(node_id: str):
    """获取特定节点的事件
    
    Path Parameters:
        node_id: 节点ID
    
    Query Parameters:
        event_type: 事件类型
        severity: 严重级别
        limit: 返回数量 (默认100)
        offset: 偏移量 (默认0)
    
    Returns:
        事件列表
    """
    event_type = request.args.get('event_type')
    severity = request.args.get('severity')
    limit = request.args.get('limit', 100, type=int)
    offset = request.args.get('offset', 0, type=int)
    
    service = get_service()
    result = service.get_node_events(
        node_id=node_id,
        event_type=event_type,
        severity=severity,
        limit=limit,
        offset=offset
    )
    
    return success_response(data=result)


@node_bp.route('/<node_id>/events', methods=['POST'])
@jwt_required()
@handle_api_errors
def create_node_event(node_id: str):
    """创建节点事件
    
    Path Parameters:
        node_id: 节点ID
    
    Request Body:
        event_type: 事件类型 (必需)
        event_message: 事件消息 (必需)
        severity: 严重级别 (info/warning/error/critical)
        source: 事件来源
    
    Returns:
        创建的事件
    """
    data = request.get_json() or {}
    
    event_type = data.get('event_type')
    event_message = data.get('event_message')
    
    if not event_type:
        return error_response('event_type is required', 400)
    if not event_message:
        return error_response('event_message is required', 400)
    
    service = get_service()
    event = service.create_event(
        node_id=node_id,
        event_type=event_type,
        event_message=event_message,
        severity=data.get('severity', 'info'),
        source=data.get('source', 'api')
    )
    
    return success_response(data=event, message='Event created'), 201


# ==================== GPU接口 ====================

@node_bp.route('/<node_id>/gpus', methods=['GET'])
@jwt_required()
@handle_api_errors
def get_node_gpus(node_id: str):
    """获取节点GPU列表
    
    Path Parameters:
        node_id: 节点ID
    
    Returns:
        GPU列表
    """
    from backend.repositories.node_repository import get_node_repository
    repo = get_node_repository()
    gpus = repo.gpus.get_gpus(node_id)
    
    return success_response(data={'gpus': gpus, 'count': len(gpus)})


@node_bp.route('/gpus/available', methods=['GET'])
@jwt_required()
@handle_api_errors
def get_available_gpus():
    """获取所有可用GPU
    
    Returns:
        可用GPU列表
    """
    from backend.repositories.node_repository import get_node_repository
    repo = get_node_repository()
    gpus = repo.gpus.get_available_gpus()
    
    return success_response(data={'gpus': gpus, 'count': len(gpus)})


@node_bp.route('/gpus/summary', methods=['GET'])
@jwt_required()
@handle_api_errors
def get_gpu_summary():
    """获取集群GPU摘要
    
    Returns:
        GPU统计摘要
    """
    from backend.repositories.node_repository import get_node_repository
    repo = get_node_repository()
    summary = repo.gpus.get_cluster_gpu_summary()
    
    return success_response(data=summary)


# ==================== 标签管理接口 ====================

@node_bp.route('/<node_id>/labels', methods=['PUT'])
@jwt_required()
@handle_api_errors
def update_node_labels(node_id: str):
    """更新节点标签
    
    Path Parameters:
        node_id: 节点ID
    
    Request Body:
        labels: 标签字典
    
    Returns:
        更新结果
    """
    data = request.get_json() or {}
    labels = data.get('labels', {})
    
    if not isinstance(labels, dict):
        return error_response('labels must be a dictionary', 400)
    
    service = get_service()
    result = service.update_node(node_id, labels=labels)
    
    if result['success']:
        return success_response(
            data={'labels': result['node'].get('labels', {})},
            message='Labels updated'
        )
    else:
        return error_response(result['message'], 400)


@node_bp.route('/<node_id>/annotations', methods=['PUT'])
@jwt_required()
@handle_api_errors
def update_node_annotations(node_id: str):
    """更新节点注解
    
    Path Parameters:
        node_id: 节点ID
    
    Request Body:
        annotations: 注解字典
    
    Returns:
        更新结果
    """
    data = request.get_json() or {}
    annotations = data.get('annotations', {})
    
    if not isinstance(annotations, dict):
        return error_response('annotations must be a dictionary', 400)
    
    service = get_service()
    result = service.update_node(node_id, annotations=annotations)
    
    if result['success']:
        return success_response(
            data={'annotations': result['node'].get('annotations', {})},
            message='Annotations updated'
        )
    else:
        return error_response(result['message'], 400)


# ==================== 服务状态接口 ====================

@node_bp.route('/service/health', methods=['GET'])
@handle_api_errors
def service_health():
    """节点服务健康检查
    
    Returns:
        服务健康状态
    """
    try:
        service = get_service()
        summary = service.get_cluster_summary()
        
        return success_response(data={
            'status': 'healthy',
            'service': 'node_service',
            'nodes_count': summary.get('nodes', {}).get('total_nodes', 0),
            'timestamp': datetime.utcnow().isoformat()
        })
    except Exception as e:
        return error_response(f'Service unhealthy: {str(e)}', 503)


@node_bp.route('/service/start', methods=['POST'])
@jwt_required()
@handle_api_errors
def start_service():
    """启动节点服务
    
    启动后台健康检查任务
    
    Returns:
        启动结果
    """
    service = get_service()
    service.start()
    
    return success_response(message='Node service started')


@node_bp.route('/service/stop', methods=['POST'])
@jwt_required()
@handle_api_errors
def stop_service():
    """停止节点服务
    
    停止后台任务
    
    Returns:
        停止结果
    """
    service = get_service()
    service.stop()
    
    return success_response(message='Node service stopped')


# ==================== 批量操作接口 ====================

@node_bp.route('/batch/cordon', methods=['POST'])
@jwt_required()
@handle_api_errors
def batch_cordon():
    """批量隔离节点
    
    Request Body:
        node_ids: 节点ID列表
        reason: 隔离原因
    
    Returns:
        批量操作结果
    """
    data = request.get_json() or {}
    node_ids = data.get('node_ids', [])
    reason = data.get('reason', 'batch operation')
    
    if not node_ids:
        return error_response('node_ids is required', 400)
    
    service = get_service()
    results = []
    
    for node_id in node_ids:
        result = service.cordon_node(node_id, reason=reason)
        results.append({
            'node_id': node_id,
            'success': result['success'],
            'message': result['message']
        })
    
    success_count = sum(1 for r in results if r['success'])
    
    return success_response(data={
        'results': results,
        'total': len(node_ids),
        'success': success_count,
        'failed': len(node_ids) - success_count
    })


@node_bp.route('/batch/uncordon', methods=['POST'])
@jwt_required()
@handle_api_errors
def batch_uncordon():
    """批量解除隔离
    
    Request Body:
        node_ids: 节点ID列表
    
    Returns:
        批量操作结果
    """
    data = request.get_json() or {}
    node_ids = data.get('node_ids', [])
    
    if not node_ids:
        return error_response('node_ids is required', 400)
    
    service = get_service()
    results = []
    
    for node_id in node_ids:
        result = service.uncordon_node(node_id)
        results.append({
            'node_id': node_id,
            'success': result['success'],
            'message': result['message']
        })
    
    success_count = sum(1 for r in results if r['success'])
    
    return success_response(data={
        'results': results,
        'total': len(node_ids),
        'success': success_count,
        'failed': len(node_ids) - success_count
    })


# ==================== 初始化和清理 ====================

def init_node_api(app=None):
    """初始化节点API
    
    Args:
        app: Flask应用实例
    """
    if app:
        app.register_blueprint(node_bp)
    
    logger.info("Node API initialized")
    return node_bp


def cleanup_node_api():
    """清理节点API资源"""
    try:
        service = get_service()
        service.stop()
        logger.info("Node API cleanup completed")
    except Exception as e:
        logger.warning(f"Node API cleanup error: {e}")

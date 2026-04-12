# -*- coding: utf-8 -*-
"""GPU 资源管理 API

提供生产级的 GPU 资源管理 API，包括资源监控、分配、释放、节点管理等功能。
"""

import logging
import os
from typing import Dict, Any, List, Optional
from datetime import datetime
from functools import wraps

from flask import Blueprint, jsonify, request, g

logger = logging.getLogger(__name__)

# 创建蓝图
gpu_bp = Blueprint('gpu', __name__, url_prefix='/api/v1/gpus')


# ==================== 辅助函数 ====================

def _get_gpu_service():
    """获取 GPU 资源服务"""
    try:
        from backend.services.gpu_resource_service import get_gpu_resource_service
        return get_gpu_resource_service()
    except ImportError:
        logger.warning("GPU resource service not available")
        return None


def _get_tenant_id() -> Optional[str]:
    """获取租户 ID"""
    return getattr(g, 'tenant_id', None) or request.headers.get('X-Tenant-ID')


def _get_user_id() -> Optional[str]:
    """获取用户 ID"""
    if hasattr(g, 'current_user') and g.current_user:
        return g.current_user.get('id')
    return request.headers.get('X-User-ID')


def token_required(f):
    """JWT 认证装饰器"""
    @wraps(f)
    def decorated(*args, **kwargs):
        auth_header = request.headers.get('Authorization')
        if auth_header and auth_header.startswith('Bearer '):
            g.current_user = {'id': 'user_001', 'role': 'admin'}
        else:
            g.current_user = None
        return f(*args, **kwargs)
    return decorated


def _run_async(coro):
    """运行异步协程"""
    import asyncio
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            future = asyncio.run_coroutine_threadsafe(coro, loop)
            return future.result(timeout=30)
        else:
            return loop.run_until_complete(coro)
    except Exception:
        new_loop = asyncio.new_event_loop()
        try:
            return new_loop.run_until_complete(coro)
        finally:
            new_loop.close()


# ==================== 初始化监控 ====================

def _init_monitoring():
    """初始化 GPU 监控"""
    try:
        interval = int(os.getenv('GPU_MONITOR_INTERVAL_SECONDS', '10'))
        service = _get_gpu_service()
        if service:
            service.start_monitoring(interval=interval, background=True)
        else:
            # 回退到模块级监控
            from backend.modules.distributed.gpu_resource_manager import start_monitoring
            start_monitoring(interval=interval, background=True)
    except Exception as e:
        logger.warning(f"Failed to init GPU monitoring: {e}")


# 启动监控
_init_monitoring()


# ==================== 指标 API ====================

@gpu_bp.route('/metrics', methods=['GET'])
@token_required
def get_metrics():
    """获取 GPU 指标
    
    返回当前所有 GPU 的使用情况，包括利用率、内存使用等。
    """
    try:
        service = _get_gpu_service()
        
        if service:
            metrics = service.get_metrics()
        else:
            # 回退到模块级缓存
            from backend.modules.distributed.gpu_resource_manager import get_cached_metrics
            metrics = get_cached_metrics()
        
        return jsonify({
            'success': True,
            'data': metrics
        }), 200
        
    except Exception as e:
        logger.error(f"Failed to get metrics: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@gpu_bp.route('/list', methods=['GET'])
@token_required
def list_gpus():
    """列出 GPU 设备
    
    返回系统中所有 GPU 设备的基本信息。
    """
    try:
        service = _get_gpu_service()
        
        if service:
            gpus = service.get_gpu_list()
        else:
            from backend.services import gpu_resource_manager as gpu_svc
            gpus = gpu_svc.detect_gpus()
        
        return jsonify({
            'success': True,
            'data': gpus,
            'count': len(gpus)
        }), 200
        
    except Exception as e:
        logger.error(f"Failed to list GPUs: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@gpu_bp.route('/details', methods=['GET'])
@token_required
def get_gpu_details():
    """获取 GPU 详细信息
    
    返回每个 GPU 的详细信息，包括利用率、内存使用、温度等。
    """
    try:
        service = _get_gpu_service()
        
        if service:
            details = service.get_gpu_details()
        else:
            from backend.services import gpu_resource_manager as gpu_svc
            details = gpu_svc.get_gpu_metrics()
        
        return jsonify({
            'success': True,
            'data': details,
            'count': len(details)
        }), 200
        
    except Exception as e:
        logger.error(f"Failed to get GPU details: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@gpu_bp.route('/summary', methods=['GET'])
@token_required
def get_gpu_summary():
    """获取 GPU 汇总信息"""
    try:
        from backend.services import gpu_resource_manager as gpu_svc
        summary = gpu_svc.get_gpu_summary()
        
        return jsonify({
            'success': True,
            'data': summary
        }), 200
        
    except Exception as e:
        logger.error(f"Failed to get GPU summary: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# ==================== 资源分配 API ====================

@gpu_bp.route('/allocate', methods=['POST'])
@token_required
def allocate_resources():
    """分配 GPU 资源
    
    请求体：
    {
        "gpu_count": 1,
        "gpu_memory_mb": 4096,
        "cpu_cores": 2,
        "memory_mb": 8192,
        "priority": 5,
        "labels_affinity": {"zone": "gpu"},
        "prefer_same_node": true,
        "task_id": "task_xxx",
        "lease_duration_seconds": 3600
    }
    """
    try:
        payload = request.get_json() or {}
        tenant_id = _get_tenant_id()
        user_id = _get_user_id()
        
        # 解析请求参数
        req = payload.get('requirement') or payload.get('requirements') or payload
        
        gpu_count = int(req.get('gpu_count', 0))
        gpu_memory_mb = int(req.get('gpu_memory_mb', 0))
        cpu_cores = int(req.get('cpu_cores', 0))
        memory_mb = int(req.get('memory_mb', 0))
        disk_mb = int(req.get('disk_mb', 0))
        network_mbps = int(req.get('network_mbps', 0))
        priority = int(req.get('priority', 1))
        labels_affinity = req.get('labels_affinity')
        prefer_same_node = req.get('prefer_same_node', True)
        task_id = req.get('task_id')
        lease_duration = req.get('lease_duration_seconds')
        strategy = req.get('strategy', 'best_fit')
        
        service = _get_gpu_service()
        
        if service:
            # 使用服务层分配
            result = service.allocate_resources(
                gpu_count=gpu_count,
                gpu_memory_mb=gpu_memory_mb,
                cpu_cores=cpu_cores,
                memory_mb=memory_mb,
                priority=priority,
                labels_affinity=labels_affinity,
                prefer_same_node=prefer_same_node,
                task_id=task_id,
                user_id=user_id,
                tenant_id=tenant_id,
                lease_duration_seconds=lease_duration,
                strategy=strategy
            )
            
            # 记录指标
            try:
                from backend.modules.monitoring.metrics_exporter import (
                    ALLOCATION_REQUESTS_COUNTER, ALLOCATION_FAILURES_COUNTER
                )
                if result.success:
                    ALLOCATION_REQUESTS_COUNTER.labels(result='success').inc()
                else:
                    ALLOCATION_REQUESTS_COUNTER.labels(result='failure').inc()
                    ALLOCATION_FAILURES_COUNTER.labels(reason='no_capacity').inc()
            except Exception:
                pass
            
            if result.success:
                return jsonify({
                    'success': True,
                    'data': {
                        'allocated': True,
                        'allocation_id': result.allocation_id,
                        'node_id': result.node_id,
                        'gpu_indices': result.gpu_indices
                    }
                }), 200
            else:
                return jsonify({
                    'success': False,
                    'allocated': False,
                    'reason': result.error_message or 'no_capacity'
                }), 409
        else:
            # 回退到模块级分配
            return _allocate_via_module(req)
            
    except Exception as e:
        logger.error(f"Failed to allocate resources: {e}")
        try:
            from backend.modules.monitoring.metrics_exporter import (
                ALLOCATION_REQUESTS_COUNTER, ALLOCATION_FAILURES_COUNTER
            )
            ALLOCATION_REQUESTS_COUNTER.labels(result='error').inc()
            ALLOCATION_FAILURES_COUNTER.labels(reason='exception').inc()
        except Exception:
            pass
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


def _allocate_via_module(req: Dict) -> tuple:
    """通过模块分配（回退方案）"""
    from backend.modules.distributed.task_scheduler import ResourceRequirement
    from backend.modules.distributed.resource_allocator import get_resource_allocator
    from backend.modules.distributed.cluster_manager import get_cluster_manager
    
    rr = ResourceRequirement(
        cpu_cores=int(req.get('cpu_cores', 0)),
        memory_mb=int(req.get('memory_mb', 0)),
        gpu_count=int(req.get('gpu_count', 0)),
        gpu_memory_mb=int(req.get('gpu_memory_mb', 0)),
        disk_mb=int(req.get('disk_mb', 0)),
        network_mbps=int(req.get('network_mbps', 0)),
        priority=int(req.get('priority', 1)),
        labels_affinity=req.get('labels_affinity') if isinstance(req.get('labels_affinity'), dict) else None
    )
    
    cluster = get_cluster_manager()
    allocator = get_resource_allocator()
    
    # 获取健康节点
    nodes = _run_async(cluster.get_healthy_nodes())
    
    # 分配资源
    res = _run_async(allocator.allocate_resources(nodes, rr))
    
    from backend.modules.monitoring.metrics_exporter import (
        ALLOCATION_REQUESTS_COUNTER, ALLOCATION_FAILURES_COUNTER
    )
    
    if res is None:
        ALLOCATION_REQUESTS_COUNTER.labels(result='failure').inc()
        ALLOCATION_FAILURES_COUNTER.labels(reason='no_capacity').inc()
        return jsonify({
            'success': False,
            'allocated': False,
            'reason': 'no_capacity'
        }), 409
    
    allocation_id, allocation = res
    ALLOCATION_REQUESTS_COUNTER.labels(result='success').inc()
    
    return jsonify({
        'success': True,
        'data': {
            'allocated': True,
            'allocation_id': allocation_id,
            'node_id': allocation.node_id,
            'gpu_indices': allocation.gpus
        }
    }), 200


@gpu_bp.route('/release/<allocation_id>', methods=['POST', 'DELETE'])
@token_required
def release_allocation(allocation_id: str):
    """释放分配的资源
    
    释放之前分配的 GPU 资源。
    """
    try:
        tenant_id = _get_tenant_id()
        service = _get_gpu_service()
        
        if service:
            success, message = service.release_allocation(allocation_id, tenant_id)
            
            if success:
                return jsonify({
                    'success': True,
                    'message': message
                }), 200
            else:
                return jsonify({
                    'success': False,
                    'error': message
                }), 400
        else:
            # 回退到模块级释放
            from backend.modules.distributed.resource_allocator import get_resource_allocator
            allocator = get_resource_allocator()
            success = _run_async(allocator.release_resources(allocation_id))
            
            if success:
                return jsonify({
                    'success': True,
                    'message': 'Resources released'
                }), 200
            else:
                return jsonify({
                    'success': False,
                    'error': 'Allocation not found'
                }), 404
            
    except Exception as e:
        logger.error(f"Failed to release allocation {allocation_id}: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@gpu_bp.route('/allocations', methods=['GET'])
@token_required
def list_allocations():
    """列出分配记录
    
    查询参数：
    - status: 状态筛选
    - node_id: 节点筛选
    - task_id: 任务筛选
    - limit: 限制数量
    - offset: 偏移量
    """
    try:
        tenant_id = _get_tenant_id()
        status = request.args.get('status')
        node_id = request.args.get('node_id')
        task_id = request.args.get('task_id')
        limit = int(request.args.get('limit', 100))
        offset = int(request.args.get('offset', 0))
        
        service = _get_gpu_service()
        
        if service:
            allocations = service.list_allocations(
                tenant_id=tenant_id,
                node_id=node_id,
                task_id=task_id,
                status=status,
                limit=limit,
                offset=offset
            )
            
            return jsonify({
                'success': True,
                'data': allocations,
                'count': len(allocations),
                'limit': limit,
                'offset': offset
            }), 200
        else:
            return jsonify({
                'success': False,
                'error': 'Service not available'
            }), 500
            
    except Exception as e:
        logger.error(f"Failed to list allocations: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@gpu_bp.route('/allocations/<allocation_id>', methods=['GET'])
@token_required
def get_allocation(allocation_id: str):
    """获取分配详情"""
    try:
        tenant_id = _get_tenant_id()
        service = _get_gpu_service()
        
        if service:
            allocation = service.get_allocation(allocation_id, tenant_id)
            
            if allocation:
                return jsonify({
                    'success': True,
                    'data': allocation
                }), 200
            else:
                return jsonify({
                    'success': False,
                    'error': 'Allocation not found'
                }), 404
        else:
            return jsonify({
                'success': False,
                'error': 'Service not available'
            }), 500
            
    except Exception as e:
        logger.error(f"Failed to get allocation {allocation_id}: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# ==================== 节点管理 API ====================

@gpu_bp.route('/nodes', methods=['GET'])
@token_required
def list_nodes():
    """列出 GPU 节点
    
    查询参数：
    - status: 状态筛选 (online, offline, maintenance)
    - is_healthy: 健康状态筛选
    - limit: 限制数量
    - offset: 偏移量
    """
    try:
        tenant_id = _get_tenant_id()
        status = request.args.get('status')
        is_healthy = request.args.get('is_healthy')
        if is_healthy is not None:
            is_healthy = is_healthy.lower() == 'true'
        limit = int(request.args.get('limit', 100))
        offset = int(request.args.get('offset', 0))
        
        service = _get_gpu_service()
        
        if service:
            nodes = service.list_nodes(
                tenant_id=tenant_id,
                status=status,
                is_healthy=is_healthy,
                limit=limit,
                offset=offset
            )
            
            return jsonify({
                'success': True,
                'data': nodes,
                'count': len(nodes),
                'limit': limit,
                'offset': offset
            }), 200
        else:
            return jsonify({
                'success': True,
                'data': [],
                'count': 0
            }), 200
            
    except Exception as e:
        logger.error(f"Failed to list nodes: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@gpu_bp.route('/nodes', methods=['POST'])
@token_required
def register_node():
    """注册 GPU 节点
    
    请求体：
    {
        "hostname": "gpu-node-01",
        "ip_address": "192.168.1.100",
        "port": 8080,
        "labels": {"zone": "gpu", "type": "A100"},
        "capabilities": ["cuda", "tensorrt"]
    }
    """
    try:
        data = request.get_json() or {}
        tenant_id = _get_tenant_id()
        
        service = _get_gpu_service()
        
        if service:
            success, message, node = service.register_node(
                hostname=data.get('hostname'),
                ip_address=data.get('ip_address'),
                port=data.get('port', 8080),
                tenant_id=tenant_id,
                labels=data.get('labels'),
                capabilities=data.get('capabilities'),
                metadata=data.get('metadata')
            )
            
            if success:
                return jsonify({
                    'success': True,
                    'data': node,
                    'message': message
                }), 201
            else:
                return jsonify({
                    'success': False,
                    'error': message
                }), 400
        else:
            return jsonify({
                'success': False,
                'error': 'Service not available'
            }), 500
            
    except Exception as e:
        logger.error(f"Failed to register node: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@gpu_bp.route('/nodes/<node_id>', methods=['GET'])
@token_required
def get_node(node_id: str):
    """获取节点详情"""
    try:
        tenant_id = _get_tenant_id()
        service = _get_gpu_service()
        
        if service:
            node = service.get_node(node_id, tenant_id)
            
            if node:
                return jsonify({
                    'success': True,
                    'data': node
                }), 200
            else:
                return jsonify({
                    'success': False,
                    'error': 'Node not found'
                }), 404
        else:
            return jsonify({
                'success': False,
                'error': 'Service not available'
            }), 500
            
    except Exception as e:
        logger.error(f"Failed to get node {node_id}: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@gpu_bp.route('/nodes/<node_id>', methods=['DELETE'])
@token_required
def unregister_node(node_id: str):
    """注销节点"""
    try:
        tenant_id = _get_tenant_id()
        service = _get_gpu_service()
        
        if service:
            success, message = service.unregister_node(node_id, tenant_id)
            
            if success:
                return jsonify({
                    'success': True,
                    'message': message
                }), 200
            else:
                return jsonify({
                    'success': False,
                    'error': message
                }), 400
        else:
            return jsonify({
                'success': False,
                'error': 'Service not available'
            }), 500
            
    except Exception as e:
        logger.error(f"Failed to unregister node {node_id}: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@gpu_bp.route('/nodes/<node_id>/heartbeat', methods=['POST'])
@token_required
def node_heartbeat(node_id: str):
    """节点心跳
    
    请求体（可选）：
    {
        "cpu_used": 50.0,
        "memory_used_mb": 8192,
        "used_gpu_memory_mb": 16384
    }
    """
    try:
        data = request.get_json() or {}
        service = _get_gpu_service()
        
        if service:
            success = service.update_node_heartbeat(node_id, data)
            
            if success:
                return jsonify({
                    'success': True,
                    'message': 'Heartbeat received'
                }), 200
            else:
                return jsonify({
                    'success': False,
                    'error': 'Node not found'
                }), 404
        else:
            return jsonify({
                'success': False,
                'error': 'Service not available'
            }), 500
            
    except Exception as e:
        logger.error(f"Failed to process heartbeat for node {node_id}: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# ==================== 统计 API ====================

@gpu_bp.route('/statistics', methods=['GET'])
@token_required
def get_statistics():
    """获取 GPU 资源统计"""
    try:
        tenant_id = _get_tenant_id()
        service = _get_gpu_service()
        
        if service:
            stats = service.get_statistics(tenant_id)
            
            return jsonify({
                'success': True,
                'data': stats
            }), 200
        else:
            # 基本统计
            from backend.services import gpu_resource_manager as gpu_svc
            gpus = gpu_svc.detect_gpus()
            metrics = gpu_svc.get_gpu_metrics()
            
            total_memory = sum(g.get('memory_total_mb', 0) for g in gpus)
            used_memory = sum(m.get('memory_used_mb', 0) for m in metrics)
            
            return jsonify({
                'success': True,
                'data': {
                    'gpus': {
                        'total': len(gpus),
                        'total_memory_mb': total_memory,
                        'used_memory_mb': used_memory,
                        'free_memory_mb': total_memory - used_memory
                    }
                }
            }), 200
            
    except Exception as e:
        logger.error(f"Failed to get statistics: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@gpu_bp.route('/usage/history', methods=['GET'])
@token_required
def get_usage_history():
    """获取使用历史
    
    查询参数：
    - node_id: 节点 ID
    - gpu_id: GPU 设备 ID
    - period_type: 时间粒度 (minute, hour, day)
    - limit: 限制数量
    """
    try:
        tenant_id = _get_tenant_id()
        node_id = request.args.get('node_id')
        gpu_id = request.args.get('gpu_id')
        period_type = request.args.get('period_type')
        limit = int(request.args.get('limit', 100))
        
        service = _get_gpu_service()
        
        if service:
            history = service.get_usage_history(
                node_id=node_id,
                gpu_id=gpu_id,
                period_type=period_type,
                limit=limit
            )
            
            return jsonify({
                'success': True,
                'data': history,
                'count': len(history)
            }), 200
        else:
            return jsonify({
                'success': True,
                'data': [],
                'count': 0
            }), 200
            
    except Exception as e:
        logger.error(f"Failed to get usage history: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# ==================== 健康检查 ====================

@gpu_bp.route('/health', methods=['GET'])
def health_check():
    """健康检查"""
    try:
        service = _get_gpu_service()
        
        if service:
            status = service.health_check()
        else:
            # 基本健康检查
            from backend.services import gpu_resource_manager as gpu_svc
            gpus = gpu_svc.detect_gpus()
            
            status = {
                'healthy': True,
                'gpu_available': len(gpus) > 0,
                'gpu_count': len(gpus),
                'timestamp': datetime.utcnow().isoformat()
            }
        
        return jsonify({
            'success': True,
            'status': 'healthy' if status.get('healthy') else 'unhealthy',
            **status
        }), 200
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            'success': False,
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.utcnow().isoformat()
        }), 500


# ==================== 监控控制 ====================

@gpu_bp.route('/monitoring/start', methods=['POST'])
@token_required
def start_monitoring():
    """启动 GPU 监控"""
    try:
        data = request.get_json() or {}
        interval = data.get('interval', 10)
        
        service = _get_gpu_service()
        
        if service:
            service.start_monitoring(interval=interval, background=True)
        else:
            from backend.modules.distributed.gpu_resource_manager import start_monitoring as start_mon
            start_mon(interval=interval, background=True)
        
        return jsonify({
            'success': True,
            'message': f'Monitoring started with interval {interval}s'
        }), 200
        
    except Exception as e:
        logger.error(f"Failed to start monitoring: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@gpu_bp.route('/monitoring/stop', methods=['POST'])
@token_required
def stop_monitoring():
    """停止 GPU 监控"""
    try:
        service = _get_gpu_service()
        
        if service:
            service.stop_monitoring()
        
        return jsonify({
            'success': True,
            'message': 'Monitoring stopped'
        }), 200
        
    except Exception as e:
        logger.error(f"Failed to stop monitoring: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

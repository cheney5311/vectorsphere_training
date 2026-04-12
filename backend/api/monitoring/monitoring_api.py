"""性能监控API接口

提供系统性能监控数据的API接口。
"""

import sys
import os
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
import psutil

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from backend.core.exceptions import ValidationError, BusinessLogicError
from backend.utils.response import success_response, error_response
from backend.modules.database.manager import get_database_manager
from backend.schemas.monitoring_models import SystemMetric
from backend.schemas.training_models import TrainingProgress

# 创建蓝图
performance_bp = Blueprint('monitoring', __name__, url_prefix='/api/v1/performance')

def get_system_metrics() -> Dict[str, Any]:
    """获取系统性能指标"""
    try:
        # CPU信息
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_count = psutil.cpu_count()
        cpu_freq = psutil.cpu_freq()
        
        # 内存信息
        memory = psutil.virtual_memory()
        
        # 磁盘信息
        disk = psutil.disk_usage('/')
        
        # 网络信息
        net_io = psutil.net_io_counters()
        
        # GPU信息（暂时不支持）
        gpu_info = []
        
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'cpu': {
                'percent': cpu_percent,
                'count': cpu_count,
                'frequency': cpu_freq.current if cpu_freq else 0
            },
            'memory': {
                'total': memory.total,
                'available': memory.available,
                'used': memory.used,
                'percent': memory.percent
            },
            'disk': {
                'total': disk.total,
                'used': disk.used,
                'free': disk.free,
                'percent': (disk.used / disk.total) * 100 if disk.total > 0 else 0
            },
            'network': {
                'bytes_sent': net_io.bytes_sent,
                'bytes_recv': net_io.bytes_recv,
                'packets_sent': net_io.packets_sent,
                'packets_recv': net_io.packets_recv
            },
            'gpu': gpu_info
        }
    except Exception as e:
        raise BusinessLogicError(f"获取系统指标失败: {str(e)}", operation="get_system_metrics")

@performance_bp.route('/system', methods=['GET'])
@jwt_required()
def get_system_performance():
    """获取系统性能数据
    
    Returns:
        {
            "timestamp": "string",
            "cpu": {
                "percent": "float",
                "count": "integer",
                "frequency": "float"
            },
            "memory": {
                "total": "integer",
                "available": "integer", 
                "used": "integer",
                "percent": "float"
            },
            "disk": {
                "total": "integer",
                "used": "integer",
                "free": "integer", 
                "percent": "float"
            },
            "network": {
                "bytes_sent": "integer",
                "bytes_recv": "integer",
                "packets_sent": "integer",
                "packets_recv": "integer"
            },
            "gpu": [
                {
                    "id": "string",
                    "name": "string", 
                    "load": "float",
                    "memory_util": "float",
                    "memory_total": "float",
                    "memory_used": "float",
                    "temperature": "float"
                }
            ]
        }
    """
    try:
        metrics = get_system_metrics()
        return success_response(metrics, "获取系统性能数据成功")
    except Exception as e:
        return error_response(f"获取系统性能数据失败: {str(e)}", 500)

@performance_bp.route('/training/<session_id>', methods=['GET'])
@jwt_required()
def get_training_performance(session_id: str):
    """获取训练任务性能数据
    
    Query Parameters:
        - start_time: 开始时间 (ISO格式)
        - end_time: 结束时间 (ISO格式)
        - limit: 限制数量 (默认: 100)
    
    Returns:
        {
            "metrics": [
                {
                    "timestamp": "string",
                    "epoch": "integer",
                    "step": "integer",
                    "loss": "float",
                    "accuracy": "float",
                    "learning_rate": "float",
                    "gpu_utilization": "float",
                    "memory_usage": "float",
                    "samples_per_second": "float"
                }
            ]
        }
    """
    try:
        # 获取查询参数
        start_time_str = request.args.get('start_time')
        end_time_str = request.args.get('end_time')
        limit = request.args.get('limit', 100, type=int)
        
        # 解析时间参数
        start_time = None
        end_time = None
        
        if start_time_str:
            try:
                start_time = datetime.fromisoformat(start_time_str)
            except ValueError:
                return error_response("无效的开始时间格式", 400)
        
        if end_time_str:
            try:
                end_time = datetime.fromisoformat(end_time_str)
            except ValueError:
                return error_response("无效的结束时间格式", 400)
        
        # 限制参数范围
        limit = min(max(limit, 1), 1000)
        
        # 获取数据库管理器
        db_manager = get_database_manager()
        
        # 从数据库获取训练指标
        with db_manager.get_db_session() as db:
            # 构建查询
            query = db.query(TrainingProgress).filter(
                TrainingProgress.session_id == session_id
            )
            
            # 根据时间范围过滤
            if start_time:
                query = query.filter(TrainingProgress.timestamp >= start_time)
            if end_time:
                query = query.filter(TrainingProgress.timestamp <= end_time)
            
            # 获取指标数据
            metrics = query.order_by(TrainingProgress.timestamp.desc()).limit(limit).all()
            
            # 转换为前端需要的格式
            metrics_data = []
            for metric in metrics:
                metric_data = {
                    'timestamp': metric.timestamp.isoformat() if metric.timestamp else None,
                    'epoch': metric.epoch,
                    'step': metric.step,
                    'loss': metric.loss,
                    'accuracy': metric.accuracy,
                    'learning_rate': metric.learning_rate,
                    'gpu_utilization': 0.0,  # TrainingProgress模型中没有这些字段
                    'memory_usage': 0.0,
                    'samples_per_second': 0.0
                }
                metrics_data.append(metric_data)
            
            # 按时间升序排列
            metrics_data.reverse()
        
        return success_response({
            'metrics': metrics_data
        }, "获取训练性能数据成功")
        
    except Exception as e:
        return error_response(f"获取训练性能数据失败: {str(e)}", 500)

@performance_bp.route('/realtime/<session_id>', methods=['GET'])
@jwt_required()
def get_realtime_performance(session_id: str):
    """获取实时性能数据（用于WebSocket推送的最新数据）
    
    Returns:
        {
            "timestamp": "string",
            "metrics": {
                "loss": "float",
                "accuracy": "float", 
                "learning_rate": "float",
                "gpu_utilization": "float",
                "memory_usage": "float",
                "samples_per_second": "float"
            }
        }
    """
    try:
        # 获取数据库管理器
        db_manager = get_database_manager()
        
        # 从数据库获取最新的训练指标
        with db_manager.get_db_session() as db:
            # 获取最新的指标数据
            latest_metric = db.query(TrainingProgress).filter(
                TrainingProgress.session_id == session_id
            ).order_by(TrainingProgress.timestamp.desc()).first()
            
            if not latest_metric:
                # 如果没有历史数据，返回系统指标
                system_metrics = get_system_metrics()
                return success_response({
                    'timestamp': system_metrics['timestamp'],
                    'metrics': {
                        'loss': 0.0,
                        'accuracy': 0.0,
                        'learning_rate': 0.0,
                        'gpu_utilization': system_metrics['cpu']['percent'],
                        'memory_usage': system_metrics['memory']['percent'],
                        'samples_per_second': 0.0
                    }
                }, "获取实时性能数据成功")
            
            metrics_data = {
                'timestamp': latest_metric.timestamp.isoformat() if latest_metric.timestamp else None,
                'metrics': {
                    'loss': latest_metric.loss,
                    'accuracy': latest_metric.accuracy,
                    'learning_rate': latest_metric.learning_rate,
                    'gpu_utilization': latest_metric.gpu_utilization or 0.0,
                    'memory_usage': latest_metric.memory_usage or 0.0,
                    'samples_per_second': latest_metric.samples_per_second or 0.0
                }
            }
        
        return success_response(metrics_data, "获取实时性能数据成功")
        
    except Exception as e:
        return error_response(f"获取实时性能数据失败: {str(e)}", 500)
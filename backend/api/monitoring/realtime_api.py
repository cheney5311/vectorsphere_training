"""实时性能监控API接口

提供实时性能监控数据的API接口，支持前端仪表盘的性能监控功能。
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
from backend.schemas.training_models import TrainingProgress, TrainingSession

# 创建蓝图
realtime_performance_bp = Blueprint('realtime_performance', __name__, url_prefix='/api/v1/realtime-performance')

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
        
        # GPU信息（暂时不支持，返回模拟数据）
        gpu_info = [{
            'id': '0',
            'name': 'NVIDIA RTX 4090',
            'load': cpu_percent,  # 使用CPU使用率作为模拟GPU负载
            'memory_util': memory.percent,
            'memory_total': memory.total / (1024**3),  # GB
            'memory_used': memory.used / (1024**3),   # GB
            'temperature': 65.0  # 模拟温度
        }]
        
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

@realtime_performance_bp.route('/system', methods=['GET'])
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

@realtime_performance_bp.route('/training/<session_id>/metrics', methods=['GET'])
@jwt_required()
def get_training_metrics(session_id: str):
    """获取训练任务性能指标数据
    
    Query Parameters:
        - limit: 限制数量 (默认: 30)
        - duration: 时间范围（分钟，默认: 15）
    
    Returns:
        {
            "metrics": [
                {
                    "timestamp": "string",
                    "gpu": {
                        "utilization": "float",
                        "memory": {
                            "used": "float",
                            "total": "float",
                            "utilization": "float"
                        },
                        "temperature": "float",
                        "powerDraw": "float"
                    },
                    "cpu": {
                        "utilization": "float",
                        "memory": {
                            "used": "float",
                            "total": "float",
                            "utilization": "float"
                        },
                        "temperature": "float"
                    },
                    "training": {
                        "samplesPerSecond": "float",
                        "tokensPerSecond": "float",
                        "batchSize": "integer",
                        "gradientNorm": "float",
                        "learningRate": "float"
                    },
                    "disk": {
                        "readSpeed": "float",
                        "writeSpeed": "float",
                        "utilization": "float"
                    },
                    "network": {
                        "downloadSpeed": "float",
                        "uploadSpeed": "float",
                        "latency": "float"
                    }
                }
            ]
        }
    """
    try:
        # 获取当前用户ID
        user_id = get_jwt_identity()
        
        # 获取查询参数
        limit = request.args.get('limit', 30, type=int)
        duration = request.args.get('duration', 15, type=int)
        
        # 限制参数范围
        limit = min(max(limit, 1), 100)
        duration = min(max(duration, 1), 60)
        
        # 计算时间范围
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(minutes=duration)
        
        # 获取数据库管理器
        db_manager = get_database_manager()
        
        # 验证训练会话所有权并获取指标数据
        with db_manager.get_db_session() as db:
            # 验证训练会话所有权
            training_session = db.query(TrainingSession).filter(
                TrainingSession.session_id == session_id,
                TrainingSession.user_id == user_id
            ).first()
            
            if not training_session:
                return error_response("训练会话不存在或无权访问", 404)
            
            # 获取训练指标数据
            metrics_query = db.query(TrainingProgress).filter(
                TrainingProgress.session_id == session_id,
                TrainingProgress.created_at >= start_time,
                TrainingProgress.created_at <= end_time
            ).order_by(TrainingProgress.created_at.desc()).limit(limit)
            
            metrics_records = metrics_query.all()
            
            # 转换为前端需要的格式
            metrics_data = []
            for record in metrics_records:
                # 获取字段值并处理None值
                gpu_utilization = float(getattr(record, 'gpu_utilization', 0) or 0)
                gpu_memory_used = float(getattr(record, 'gpu_memory_used', 0) or 0)
                gpu_memory_total = float(getattr(record, 'gpu_memory_total', 0) or 0)
                gpu_temperature = float(getattr(record, 'gpu_temperature', 0) or 0)
                gpu_power_draw = float(getattr(record, 'gpu_power_draw', 0) or 0)
                
                cpu_utilization = float(getattr(record, 'cpu_utilization', 0) or 0)
                cpu_memory_used = float(getattr(record, 'cpu_memory_used', 0) or 0)
                cpu_memory_total = float(getattr(record, 'cpu_memory_total', 0) or 0)
                cpu_temperature = float(getattr(record, 'cpu_temperature', 0) or 0)
                
                samples_per_second = float(getattr(record, 'samples_per_second', 0) or 0)
                tokens_per_second = float(getattr(record, 'tokens_per_second', 0) or 0)
                batch_size = int(getattr(record, 'batch_size', 0) or 0)
                gradient_norm = float(getattr(record, 'gradient_norm', 0) or 0)
                learning_rate = float(getattr(record, 'learning_rate', 0) or 0)
                
                disk_read_speed = float(getattr(record, 'disk_read_speed', 0) or 0)
                disk_write_speed = float(getattr(record, 'disk_write_speed', 0) or 0)
                disk_utilization = float(getattr(record, 'disk_utilization', 0) or 0)
                
                network_download_speed = float(getattr(record, 'network_download_speed', 0) or 0)
                network_upload_speed = float(getattr(record, 'network_upload_speed', 0) or 0)
                network_latency = float(getattr(record, 'network_latency', 0) or 0)
                
                # 处理GPU内存使用率计算
                gpu_memory_utilization = 0.0
                if gpu_memory_total is not None and gpu_memory_total > 0:
                    gpu_memory_utilization = (gpu_memory_used / gpu_memory_total * 100)
                
                # 处理CPU内存使用率计算
                cpu_memory_utilization = 0.0
                if cpu_memory_total is not None and cpu_memory_total > 0:
                    cpu_memory_utilization = (cpu_memory_used / cpu_memory_total * 100)
                
                metric_data = {
                    'timestamp': record.created_at.isoformat() if record.created_at else datetime.utcnow().isoformat(),
                    'gpu': {
                        'utilization': gpu_utilization,
                        'memory': {
                            'used': gpu_memory_used,
                            'total': gpu_memory_total,
                            'utilization': gpu_memory_utilization
                        },
                        'temperature': gpu_temperature,
                        'powerDraw': gpu_power_draw
                    },
                    'cpu': {
                        'utilization': cpu_utilization,
                        'memory': {
                            'used': cpu_memory_used,
                            'total': cpu_memory_total,
                            'utilization': cpu_memory_utilization
                        },
                        'temperature': cpu_temperature
                    },
                    'training': {
                        'samplesPerSecond': samples_per_second,
                        'tokensPerSecond': tokens_per_second,
                        'batchSize': batch_size,
                        'gradientNorm': gradient_norm,
                        'learningRate': learning_rate
                    },
                    'disk': {
                        'readSpeed': disk_read_speed,
                        'writeSpeed': disk_write_speed,
                        'utilization': disk_utilization
                    },
                    'network': {
                        'downloadSpeed': network_download_speed,
                        'uploadSpeed': network_upload_speed,
                        'latency': network_latency
                    }
                }
                metrics_data.append(metric_data)
            
            # 按时间升序排列
            metrics_data.reverse()
        
        return success_response({
            'metrics': metrics_data
        }, "获取训练性能指标成功")
        
    except Exception as e:
        return error_response(f"获取训练性能指标失败: {str(e)}", 500)

@realtime_performance_bp.route('/training/<session_id>/current', methods=['GET'])
@jwt_required()
def get_current_training_metrics(session_id: str):
    """获取训练任务当前性能指标数据
    
    Returns:
        {
            "timestamp": "string",
            "gpu": {
                "utilization": "float",
                "memory": {
                    "used": "float",
                    "total": "float",
                    "utilization": "float"
                },
                "temperature": "float",
                "powerDraw": "float"
            },
            "cpu": {
                "utilization": "float",
                "memory": {
                    "used": "float",
                    "total": "float",
                    "utilization": "float"
                },
                "temperature": "float"
            },
            "training": {
                "samplesPerSecond": "float",
                "tokensPerSecond": "float",
                "batchSize": "integer",
                "gradientNorm": "float",
                "learningRate": "float"
            },
            "disk": {
                "readSpeed": "float",
                "writeSpeed": "float",
                "utilization": "float"
            },
            "network": {
                "downloadSpeed": "float",
                "uploadSpeed": "float",
                "latency": "float"
            }
        }
    """
    try:
        # 获取当前用户ID
        user_id = get_jwt_identity()
        
        # 获取数据库管理器
        db_manager = get_database_manager()
        
        # 验证训练会话所有权并获取最新指标数据
        with db_manager.get_db_session() as db:
            # 验证训练会话所有权
            training_session = db.query(TrainingSession).filter(
                TrainingSession.session_id == session_id,
                TrainingSession.user_id == user_id
            ).first()
            
            if not training_session:
                return error_response("训练会话不存在或无权访问", 404)
            
            # 获取最新的训练指标数据
            latest_metric = db.query(TrainingProgress).filter(
                TrainingProgress.session_id == session_id
            ).order_by(TrainingProgress.created_at.desc()).first()
            
            if not latest_metric:
                # 如果没有历史数据，返回系统指标
                system_metrics = get_system_metrics()
                current_metric = {
                    'timestamp': system_metrics['timestamp'],
                    'gpu': {
                        'utilization': system_metrics['cpu']['percent'],  # 使用CPU使用率作为模拟GPU负载
                        'memory': {
                            'used': system_metrics['memory']['used'] / (1024**3),
                            'total': system_metrics['memory']['total'] / (1024**3),
                            'utilization': system_metrics['memory']['percent']
                        },
                        'temperature': 65.0,  # 模拟温度
                        'powerDraw': 150.0   # 模拟功耗
                    },
                    'cpu': {
                        'utilization': system_metrics['cpu']['percent'],
                        'memory': {
                            'used': system_metrics['memory']['used'] / (1024**3),
                            'total': system_metrics['memory']['total'] / (1024**3),
                            'utilization': system_metrics['memory']['percent']
                        },
                        'temperature': 45.0  # 模拟温度
                    },
                    'training': {
                        'samplesPerSecond': 0.0,
                        'tokensPerSecond': 0.0,
                        'batchSize': 0,
                        'gradientNorm': 0.0,
                        'learningRate': 0.0
                    },
                    'disk': {
                        'readSpeed': 0.0,
                        'writeSpeed': 0.0,
                        'utilization': system_metrics['disk']['percent']
                    },
                    'network': {
                        'downloadSpeed': 0.0,
                        'uploadSpeed': 0.0,
                        'latency': 0.0
                    }
                }
            else:
                # 获取字段值并处理None值
                gpu_utilization = float(getattr(latest_metric, 'gpu_utilization', 0) or 0)
                gpu_memory_used = float(getattr(latest_metric, 'gpu_memory_used', 0) or 0)
                gpu_memory_total = float(getattr(latest_metric, 'gpu_memory_total', 0) or 0)
                gpu_temperature = float(getattr(latest_metric, 'gpu_temperature', 0) or 0)
                gpu_power_draw = float(getattr(latest_metric, 'gpu_power_draw', 0) or 0)
                
                cpu_utilization = float(getattr(latest_metric, 'cpu_utilization', 0) or 0)
                cpu_memory_used = float(getattr(latest_metric, 'cpu_memory_used', 0) or 0)
                cpu_memory_total = float(getattr(latest_metric, 'cpu_memory_total', 0) or 0)
                cpu_temperature = float(getattr(latest_metric, 'cpu_temperature', 0) or 0)
                
                samples_per_second = float(getattr(latest_metric, 'samples_per_second', 0) or 0)
                tokens_per_second = float(getattr(latest_metric, 'tokens_per_second', 0) or 0)
                batch_size = int(getattr(latest_metric, 'batch_size', 0) or 0)
                gradient_norm = float(getattr(latest_metric, 'gradient_norm', 0) or 0)
                learning_rate = float(getattr(latest_metric, 'learning_rate', 0) or 0)
                
                disk_read_speed = float(getattr(latest_metric, 'disk_read_speed', 0) or 0)
                disk_write_speed = float(getattr(latest_metric, 'disk_write_speed', 0) or 0)
                disk_utilization = float(getattr(latest_metric, 'disk_utilization', 0) or 0)
                
                network_download_speed = float(getattr(latest_metric, 'network_download_speed', 0) or 0)
                network_upload_speed = float(getattr(latest_metric, 'network_upload_speed', 0) or 0)
                network_latency = float(getattr(latest_metric, 'network_latency', 0) or 0)
                
                # 处理GPU内存使用率计算
                gpu_memory_utilization = 0.0
                if gpu_memory_total is not None and gpu_memory_total > 0:
                    gpu_memory_utilization = (gpu_memory_used / gpu_memory_total * 100)
                
                # 处理CPU内存使用率计算
                cpu_memory_utilization = 0.0
                if cpu_memory_total is not None and cpu_memory_total > 0:
                    cpu_memory_utilization = (cpu_memory_used / cpu_memory_total * 100)
                
                current_metric = {
                    'timestamp': latest_metric.created_at.isoformat() if latest_metric.created_at else datetime.utcnow().isoformat(),
                    'gpu': {
                        'utilization': gpu_utilization,
                        'memory': {
                            'used': gpu_memory_used,
                            'total': gpu_memory_total,
                            'utilization': gpu_memory_utilization
                        },
                        'temperature': gpu_temperature,
                        'powerDraw': gpu_power_draw
                    },
                    'cpu': {
                        'utilization': cpu_utilization,
                        'memory': {
                            'used': cpu_memory_used,
                            'total': cpu_memory_total,
                            'utilization': cpu_memory_utilization
                        },
                        'temperature': cpu_temperature
                    },
                    'training': {
                        'samplesPerSecond': samples_per_second,
                        'tokensPerSecond': tokens_per_second,
                        'batchSize': batch_size,
                        'gradientNorm': gradient_norm,
                        'learningRate': learning_rate
                    },
                    'disk': {
                        'readSpeed': disk_read_speed,
                        'writeSpeed': disk_write_speed,
                        'utilization': disk_utilization
                    },
                    'network': {
                        'downloadSpeed': network_download_speed,
                        'uploadSpeed': network_upload_speed,
                        'latency': network_latency
                    }
                }
        
        return success_response(current_metric, "获取当前训练性能指标成功")
        
    except Exception as e:
        return error_response(f"获取当前训练性能指标失败: {str(e)}", 500)

@realtime_performance_bp.route('/training/<session_id>/stats', methods=['GET'])
@jwt_required()
def get_training_statistics(session_id: str):
    """获取训练任务统计信息
    
    Returns:
        {
            "avgGpuUtilization": "float",
            "avgGpuUsage": "float",
            "avgMemoryUsage": "float",
            "maxGpuMemory": "float",
            "avgTrainingSpeed": "float",
            "peakTemperature": "float",
            "totalSamplesProcessed": "integer",
            "totalPowerConsumption": "float",
            "uptime": "integer"
        }
    """
    try:
        # 获取当前用户ID
        user_id = get_jwt_identity()
        
        # 获取数据库管理器
        db_manager = get_database_manager()
        
        # 验证训练会话所有权并获取统计信息
        with db_manager.get_db_session() as db:
            # 验证训练会话所有权
            training_session = db.query(TrainingSession).filter(
                TrainingSession.session_id == session_id,
                TrainingSession.user_id == user_id
            ).first()
            
            if not training_session:
                return error_response("训练会话不存在或无权访问", 404)
            
            # 获取所有训练指标数据用于统计
            metrics_records = db.query(TrainingProgress).filter(
                TrainingProgress.session_id == session_id
            ).order_by(TrainingProgress.created_at.asc()).all()
            
            if not metrics_records:
                # 如果没有历史数据，返回默认统计信息
                stats = {
                    'avgGpuUtilization': 0.0,
                    'avgGpuUsage': 0.0,
                    'avgMemoryUsage': 0.0,
                    'maxGpuMemory': 0.0,
                    'avgTrainingSpeed': 0.0,
                    'peakTemperature': 0.0,
                    'totalSamplesProcessed': 0,
                    'totalPowerConsumption': 0.0,
                    'uptime': 0
                }
            else:
                # 提取非空值用于统计计算
                gpu_utilizations = [float(getattr(m, 'gpu_utilization', 0) or 0) for m in metrics_records if getattr(m, 'gpu_utilization', None) is not None]
                cpu_utilizations = [float(getattr(m, 'cpu_utilization', 0) or 0) for m in metrics_records if getattr(m, 'cpu_utilization', None) is not None]
                gpu_memory_totals = [float(getattr(m, 'gpu_memory_total', 0) or 0) for m in metrics_records if getattr(m, 'gpu_memory_total', None) is not None]
                samples_per_seconds = [float(getattr(m, 'samples_per_second', 0) or 0) for m in metrics_records if getattr(m, 'samples_per_second', None) is not None]
                gpu_temperatures = [float(getattr(m, 'gpu_temperature', 0) or 0) for m in metrics_records if getattr(m, 'gpu_temperature', None) is not None]
                gpu_power_draws = [float(getattr(m, 'gpu_power_draw', 0) or 0) for m in metrics_records if getattr(m, 'gpu_power_draw', None) is not None]
                batch_sizes = [int(getattr(m, 'batch_size', 0) or 0) for m in metrics_records if getattr(m, 'batch_size', None) is not None]
                
                # 计算运行时间（秒）
                uptime = 0
                started_at = getattr(training_session, 'started_at', None)
                completed_at = getattr(training_session, 'completed_at', None)
                if started_at is not None:
                    if completed_at is not None:
                        uptime = (completed_at - started_at).total_seconds()
                    else:
                        uptime = (datetime.utcnow() - started_at).total_seconds()
                
                # 计算总样本数
                total_samples = sum(batch_sizes) if batch_sizes else 0
                
                # 计算统计值
                avg_gpu_utilization = sum(gpu_utilizations) / len(gpu_utilizations) if gpu_utilizations else 0.0
                avg_memory_usage = sum(cpu_utilizations) / len(cpu_utilizations) if cpu_utilizations else 0.0
                max_gpu_memory = max(gpu_memory_totals) if gpu_memory_totals else 0.0
                avg_training_speed = sum(samples_per_seconds) / len(samples_per_seconds) if samples_per_seconds else 0.0
                peak_temperature = max(gpu_temperatures) if gpu_temperatures else 0.0
                total_power_consumption = sum(gpu_power_draws) if gpu_power_draws else 0.0
                
                stats = {
                    'avgGpuUtilization': avg_gpu_utilization,
                    'avgGpuUsage': avg_gpu_utilization,  # 兼容前端代码
                    'avgMemoryUsage': avg_memory_usage,
                    'maxGpuMemory': max_gpu_memory,
                    'avgTrainingSpeed': avg_training_speed,
                    'peakTemperature': peak_temperature,
                    'totalSamplesProcessed': total_samples,
                    'totalPowerConsumption': total_power_consumption,
                    'uptime': int(uptime)
                }
        
        return success_response(stats, "获取训练统计信息成功")
        
    except Exception as e:
        return error_response(f"获取训练统计信息失败: {str(e)}", 500)
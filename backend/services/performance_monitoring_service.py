#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""性能监控业务逻辑层

提供性能监控的核心业务逻辑：
- 系统性能指标采集和分析
- GPU性能监控
- 训练任务性能监控
- 性能告警管理
- 性能统计和分析

整合 monitoring 模块的功能到 performance 模块中。

架构调用关系：
API层 (monitoring_api.py / performance_api.py)
    -> Service层 (本模块)
        -> Repository层 (performance_monitoring_repository.py)
"""

import logging
import psutil
import threading
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class PerformanceMonitoringService:
    """性能监控业务逻辑层
    
    整合系统监控、GPU监控、训练监控功能，
    提供完整的性能监控业务逻辑。
    """
    
    # 资源阈值配置
    THRESHOLDS = {
        'cpu': {'warning': 80, 'critical': 90},
        'memory': {'warning': 75, 'critical': 85},
        'disk': {'warning': 80, 'critical': 90},
        'gpu_utilization': {'warning': 80, 'critical': 95},
        'gpu_temperature': {'warning': 75, 'critical': 85},
    }
    
    def __init__(self, use_memory: bool = False):
        """初始化服务
        
        Args:
            use_memory: 是否使用内存存储（用于测试）
        """
        self._use_memory = use_memory
        self._lock = threading.RLock()
        self._repository = None
        self._collection_status = 'stopped'
        self._collection_interval = 10  # 采集间隔（秒）
        
        self._init_dependencies()
    
    def _init_dependencies(self):
        """初始化依赖"""
        try:
            from backend.repositories.performance_monitoring_repository import get_monitoring_repository
            self._repository = get_monitoring_repository(use_memory=self._use_memory)
            logger.info("PerformanceMonitoringService: Repository initialized")
        except Exception as e:
            logger.warning(f"PerformanceMonitoringService: Failed to init repository: {e}")
    
    # ==========================================================================
    # 系统性能监控
    # ==========================================================================
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """获取当前系统性能指标
        
        Returns:
            系统性能指标，包含 CPU、内存、磁盘、网络信息
        """
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
            
            # 构建指标数据
            metrics = {
                'timestamp': datetime.utcnow().isoformat(),
                'cpu': {
                    'percent': cpu_percent,
                    'count': cpu_count,
                    'frequency': cpu_freq.current if cpu_freq else 0,
                    'status': self._assess_status(cpu_percent, 'cpu'),
                },
                'memory': {
                    'total': memory.total,
                    'available': memory.available,
                    'used': memory.used,
                    'percent': memory.percent,
                    'status': self._assess_status(memory.percent, 'memory'),
                },
                'disk': {
                    'total': disk.total,
                    'used': disk.used,
                    'free': disk.free,
                    'percent': (disk.used / disk.total) * 100 if disk.total > 0 else 0,
                    'status': self._assess_status((disk.used / disk.total) * 100 if disk.total > 0 else 0, 'disk'),
                },
                'network': {
                    'bytes_sent': net_io.bytes_sent,
                    'bytes_recv': net_io.bytes_recv,
                    'packets_sent': net_io.packets_sent,
                    'packets_recv': net_io.packets_recv,
                },
                'gpu': self._get_gpu_info(),
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to get system metrics: {e}")
            raise
    
    def _get_gpu_info(self) -> List[Dict[str, Any]]:
        """获取GPU信息"""
        gpu_info = []
        
        try:
            # 尝试使用 pynvml 获取 NVIDIA GPU 信息
            import pynvml
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                name = pynvml.nvmlDeviceGetName(handle)
                if isinstance(name, bytes):
                    name = name.decode('utf-8')
                
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000  # mW to W
                
                gpu_info.append({
                    'id': str(i),
                    'name': name,
                    'load': util.gpu,
                    'utilization': util.gpu,
                    'memory_util': (memory.used / memory.total) * 100 if memory.total > 0 else 0,
                    'memory_total': memory.total / (1024**3),  # GB
                    'memory_used': memory.used / (1024**3),    # GB
                    'memory_free': memory.free / (1024**3),    # GB
                    'temperature': temp,
                    'powerDraw': power,
                    'status': self._assess_status(util.gpu, 'gpu_utilization'),
                })
            
            pynvml.nvmlShutdown()
            
        except ImportError:
            logger.debug("pynvml not available, using simulated GPU data")
            # 使用模拟GPU数据
            memory = psutil.virtual_memory()
            gpu_info = [{
                'id': '0',
                'name': 'Simulated GPU',
                'load': psutil.cpu_percent(),
                'utilization': psutil.cpu_percent(),
                'memory_util': memory.percent,
                'memory_total': memory.total / (1024**3),
                'memory_used': memory.used / (1024**3),
                'memory_free': memory.available / (1024**3),
                'temperature': 65.0,
                'powerDraw': 150.0,
                'status': 'healthy',
            }]
        except Exception as e:
            logger.warning(f"Failed to get GPU info: {e}")
            # 返回空列表或模拟数据
            gpu_info = []
        
        return gpu_info
    
    def _assess_status(self, value: float, metric_type: str) -> str:
        """评估资源状态
        
        Args:
            value: 当前值
            metric_type: 指标类型
            
        Returns:
            状态字符串: healthy/warning/critical
        """
        thresholds = self.THRESHOLDS.get(metric_type, {'warning': 80, 'critical': 90})
        
        if value >= thresholds['critical']:
            return 'critical'
        elif value >= thresholds['warning']:
            return 'warning'
        return 'healthy'
    
    def record_system_metrics(self, tenant_id: Optional[str] = None) -> Dict[str, Any]:
        """记录系统性能指标
        
        Args:
            tenant_id: 租户ID
            
        Returns:
            记录的指标数据
        """
        metrics = self.get_system_metrics()
        
        if self._repository:
            return self._repository.record_system_metrics(metrics, tenant_id)
        
        return metrics
    
    def get_system_metrics_history(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100,
        tenant_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """获取系统指标历史
        
        Args:
            start_time: 开始时间
            end_time: 结束时间
            limit: 限制数量
            tenant_id: 租户ID
            
        Returns:
            历史指标数据
        """
        if self._repository:
            records, total = self._repository.get_system_metrics_history(
                start_time=start_time,
                end_time=end_time,
                limit=limit,
                tenant_id=tenant_id
            )
            return {
                'metrics': records,
                'total': total,
                'limit': limit,
            }
        
        return {'metrics': [], 'total': 0, 'limit': limit}
    
    # ==========================================================================
    # 训练性能监控
    # ==========================================================================
    
    def get_training_metrics(
        self,
        session_id: str,
        user_id: str,
        limit: int = 30,
        duration_minutes: int = 15
    ) -> Dict[str, Any]:
        """获取训练任务性能指标
        
        Args:
            session_id: 训练会话ID
            user_id: 用户ID
            limit: 限制数量
            duration_minutes: 时间范围（分钟）
            
        Returns:
            训练性能指标
        """
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(minutes=duration_minutes)
        
        if self._repository:
            records, total = self._repository.get_training_metrics_history(
                session_id=session_id,
                start_time=start_time,
                end_time=end_time,
                limit=limit
            )
            
            # 按时间升序排列
            records.reverse()
            
            return {
                'metrics': records,
                'total': total,
            }
        
        return {'metrics': [], 'total': 0}
    
    def get_current_training_metrics(
        self,
        session_id: str,
        user_id: str
    ) -> Dict[str, Any]:
        """获取训练任务当前性能指标
        
        Args:
            session_id: 训练会话ID
            user_id: 用户ID
            
        Returns:
            当前训练性能指标
        """
        if self._repository:
            latest = self._repository.get_latest_training_metrics(session_id)
            if latest:
                return latest
        
        # 返回默认/实时系统指标
        system_metrics = self.get_system_metrics()
        
        return {
            'timestamp': system_metrics['timestamp'],
            'gpu': {
                'utilization': system_metrics['cpu']['percent'],
                'memory': {
                    'used': system_metrics['memory']['used'] / (1024**3),
                    'total': system_metrics['memory']['total'] / (1024**3),
                    'utilization': system_metrics['memory']['percent'],
                },
                'temperature': 65.0,
                'powerDraw': 150.0,
            },
            'cpu': {
                'utilization': system_metrics['cpu']['percent'],
                'memory': {
                    'used': system_metrics['memory']['used'] / (1024**3),
                    'total': system_metrics['memory']['total'] / (1024**3),
                    'utilization': system_metrics['memory']['percent'],
                },
                'temperature': 45.0,
            },
            'training': {
                'samplesPerSecond': 0.0,
                'tokensPerSecond': 0.0,
                'batchSize': 0,
                'gradientNorm': 0.0,
                'learningRate': 0.0,
            },
            'disk': {
                'readSpeed': 0.0,
                'writeSpeed': 0.0,
                'utilization': system_metrics['disk']['percent'],
            },
            'network': {
                'downloadSpeed': 0.0,
                'uploadSpeed': 0.0,
                'latency': 0.0,
            },
        }
    
    def get_training_statistics(
        self,
        session_id: str,
        user_id: str
    ) -> Dict[str, Any]:
        """获取训练任务统计信息
        
        Args:
            session_id: 训练会话ID
            user_id: 用户ID
            
        Returns:
            训练统计信息
        """
        if self._repository:
            stats = self._repository.get_training_statistics(session_id)
            return {
                'avgGpuUtilization': stats.get('avg_gpu_utilization', 0.0),
                'avgGpuUsage': stats.get('avg_gpu_utilization', 0.0),
                'avgMemoryUsage': stats.get('avg_memory_usage', 0.0),
                'maxGpuMemory': stats.get('max_gpu_memory', 0.0),
                'avgTrainingSpeed': stats.get('avg_training_speed', 0.0),
                'peakTemperature': stats.get('peak_temperature', 0.0),
                'totalSamplesProcessed': stats.get('total_samples_processed', 0),
                'totalPowerConsumption': stats.get('total_power_consumption', 0.0),
                'uptime': stats.get('uptime_seconds', 0),
            }
        
        return {
            'avgGpuUtilization': 0.0,
            'avgGpuUsage': 0.0,
            'avgMemoryUsage': 0.0,
            'maxGpuMemory': 0.0,
            'avgTrainingSpeed': 0.0,
            'peakTemperature': 0.0,
            'totalSamplesProcessed': 0,
            'totalPowerConsumption': 0.0,
            'uptime': 0,
        }
    
    def record_training_metrics(
        self,
        session_id: str,
        user_id: str,
        metrics: Dict[str, Any],
        tenant_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """记录训练性能指标
        
        Args:
            session_id: 训练会话ID
            user_id: 用户ID
            metrics: 指标数据
            tenant_id: 租户ID
            
        Returns:
            记录的指标数据
        """
        if self._repository:
            return self._repository.record_training_metrics(
                session_id=session_id,
                user_id=user_id,
                metrics=metrics,
                tenant_id=tenant_id
            )
        
        return metrics
    
    # ==========================================================================
    # 性能告警管理
    # ==========================================================================
    
    def check_and_create_alerts(self, metrics: Dict[str, Any], session_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """检查指标并创建告警
        
        Args:
            metrics: 性能指标
            session_id: 训练会话ID
            
        Returns:
            创建的告警列表
        """
        alerts = []
        
        # 检查CPU
        cpu_percent = metrics.get('cpu', {}).get('percent', 0)
        if cpu_percent >= self.THRESHOLDS['cpu']['critical']:
            alert = self._create_alert(
                alert_type='cpu_critical',
                level='critical',
                title='CPU使用率严重过高',
                message=f'CPU使用率达到 {cpu_percent:.1f}%，超过严重阈值 {self.THRESHOLDS["cpu"]["critical"]}%',
                source='system',
                metric_name='cpu_percent',
                metric_value=cpu_percent,
                threshold=self.THRESHOLDS['cpu']['critical'],
                session_id=session_id
            )
            if alert:
                alerts.append(alert)
        elif cpu_percent >= self.THRESHOLDS['cpu']['warning']:
            alert = self._create_alert(
                alert_type='cpu_warning',
                level='warning',
                title='CPU使用率过高',
                message=f'CPU使用率达到 {cpu_percent:.1f}%，超过警告阈值 {self.THRESHOLDS["cpu"]["warning"]}%',
                source='system',
                metric_name='cpu_percent',
                metric_value=cpu_percent,
                threshold=self.THRESHOLDS['cpu']['warning'],
                session_id=session_id
            )
            if alert:
                alerts.append(alert)
        
        # 检查内存
        memory_percent = metrics.get('memory', {}).get('percent', 0)
        if memory_percent >= self.THRESHOLDS['memory']['critical']:
            alert = self._create_alert(
                alert_type='memory_critical',
                level='critical',
                title='内存使用率严重过高',
                message=f'内存使用率达到 {memory_percent:.1f}%，超过严重阈值 {self.THRESHOLDS["memory"]["critical"]}%',
                source='system',
                metric_name='memory_percent',
                metric_value=memory_percent,
                threshold=self.THRESHOLDS['memory']['critical'],
                session_id=session_id
            )
            if alert:
                alerts.append(alert)
        
        # 检查磁盘
        disk_percent = metrics.get('disk', {}).get('percent', 0)
        if disk_percent >= self.THRESHOLDS['disk']['critical']:
            alert = self._create_alert(
                alert_type='disk_critical',
                level='critical',
                title='磁盘使用率严重过高',
                message=f'磁盘使用率达到 {disk_percent:.1f}%，超过严重阈值 {self.THRESHOLDS["disk"]["critical"]}%',
                source='system',
                metric_name='disk_percent',
                metric_value=disk_percent,
                threshold=self.THRESHOLDS['disk']['critical'],
                session_id=session_id
            )
            if alert:
                alerts.append(alert)
        
        # 检查GPU
        for gpu in metrics.get('gpu', []):
            gpu_util = gpu.get('utilization') or gpu.get('load', 0)
            gpu_temp = gpu.get('temperature', 0)
            gpu_id = gpu.get('id', '0')
            
            if gpu_temp >= self.THRESHOLDS['gpu_temperature']['critical']:
                alert = self._create_alert(
                    alert_type='gpu_temperature_critical',
                    level='critical',
                    title=f'GPU {gpu_id} 温度严重过高',
                    message=f'GPU {gpu_id} 温度达到 {gpu_temp}°C，超过严重阈值 {self.THRESHOLDS["gpu_temperature"]["critical"]}°C',
                    source='gpu',
                    metric_name='gpu_temperature',
                    metric_value=gpu_temp,
                    threshold=self.THRESHOLDS['gpu_temperature']['critical'],
                    session_id=session_id
                )
                if alert:
                    alerts.append(alert)
        
        return alerts
    
    def _create_alert(
        self,
        alert_type: str,
        level: str,
        title: str,
        message: str,
        source: str,
        metric_name: Optional[str] = None,
        metric_value: Optional[float] = None,
        threshold: Optional[float] = None,
        session_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """创建告警"""
        if self._repository:
            return self._repository.create_alert(
                alert_type=alert_type,
                level=level,
                title=title,
                message=message,
                source=source,
                metric_name=metric_name,
                metric_value=metric_value,
                threshold=threshold,
                session_id=session_id
            )
        return None
    
    def get_active_alerts(self, limit: int = 100) -> List[Dict[str, Any]]:
        """获取活动告警"""
        if self._repository:
            alerts, _ = self._repository.list_alerts(status='active', limit=limit)
            return alerts
        return []
    
    def acknowledge_alert(self, alert_id: str, user_id: str) -> Optional[Dict[str, Any]]:
        """确认告警"""
        if self._repository:
            return self._repository.acknowledge_alert(alert_id, user_id)
        return None
    
    def resolve_alert(self, alert_id: str, user_id: str, resolution_note: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """解决告警"""
        if self._repository:
            return self._repository.resolve_alert(alert_id, user_id, resolution_note)
        return None
    
    def get_alert_statistics(self) -> Dict[str, Any]:
        """获取告警统计"""
        if self._repository:
            return self._repository.get_alert_statistics()
        return {
            'total': 0,
            'by_level': {},
            'by_status': {},
            'by_type': {},
            'active_count': 0,
            'resolved_count': 0,
        }
    
    # ==========================================================================
    # 采集控制
    # ==========================================================================
    
    def start_collection(self, interval: int = 10) -> Dict[str, Any]:
        """启动性能指标采集
        
        Args:
            interval: 采集间隔（秒）
            
        Returns:
            采集状态
        """
        with self._lock:
            self._collection_status = 'running'
            self._collection_interval = interval
        
        logger.info(f"Performance collection started with interval {interval}s")
        
        return {
            'status': 'running',
            'interval': interval,
            'message': '性能指标采集已启动',
        }
    
    def stop_collection(self) -> Dict[str, Any]:
        """停止性能指标采集"""
        with self._lock:
            self._collection_status = 'stopped'
        
        logger.info("Performance collection stopped")
        
        return {
            'status': 'stopped',
            'message': '性能指标采集已停止',
        }
    
    def get_collection_status(self) -> Dict[str, Any]:
        """获取采集状态"""
        with self._lock:
            return {
                'status': self._collection_status,
                'interval': self._collection_interval,
            }
    
    # ==========================================================================
    # 健康检查
    # ==========================================================================
    
    def get_service_health(self) -> Dict[str, Any]:
        """获取服务健康状态"""
        try:
            metrics = self.get_system_metrics()
            
            # 评估整体健康状态
            cpu_status = metrics['cpu'].get('status', 'healthy')
            memory_status = metrics['memory'].get('status', 'healthy')
            disk_status = metrics['disk'].get('status', 'healthy')
            
            status_priority = {'critical': 3, 'warning': 2, 'healthy': 1}
            max_status = max([cpu_status, memory_status, disk_status], 
                           key=lambda x: status_priority.get(x, 0))
            
            return {
                'status': max_status,
                'timestamp': datetime.utcnow().isoformat(),
                'components': {
                    'cpu': cpu_status,
                    'memory': memory_status,
                    'disk': disk_status,
                },
                'metrics': {
                    'cpu_percent': metrics['cpu']['percent'],
                    'memory_percent': metrics['memory']['percent'],
                    'disk_percent': metrics['disk']['percent'],
                },
                'collection': self.get_collection_status(),
            }
        except Exception as e:
            logger.error(f"Failed to get service health: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat(),
            }
    
    # ==========================================================================
    # 监控摘要
    # ==========================================================================
    
    def get_monitoring_summary(self, time_range: str = '1h') -> Dict[str, Any]:
        """获取监控摘要
        
        Args:
            time_range: 时间范围
            
        Returns:
            监控摘要数据
        """
        # 解析时间范围
        time_ranges = {
            '5m': timedelta(minutes=5),
            '15m': timedelta(minutes=15),
            '30m': timedelta(minutes=30),
            '1h': timedelta(hours=1),
            '6h': timedelta(hours=6),
            '24h': timedelta(hours=24),
        }
        duration = time_ranges.get(time_range, timedelta(hours=1))
        
        end_time = datetime.utcnow()
        start_time = end_time - duration
        
        # 获取当前指标
        current_metrics = self.get_system_metrics()
        
        # 获取历史指标
        history = self.get_system_metrics_history(
            start_time=start_time,
            end_time=end_time,
            limit=1000
        )
        
        # 获取告警统计
        alert_stats = self.get_alert_statistics()
        
        return {
            'timestamp': current_metrics['timestamp'],
            'timeRange': time_range,
            'current': {
                'cpu': current_metrics['cpu'],
                'memory': current_metrics['memory'],
                'disk': current_metrics['disk'],
                'network': current_metrics['network'],
                'gpu': current_metrics['gpu'],
            },
            'history': {
                'count': history['total'],
            },
            'alerts': alert_stats,
            'health': self.get_service_health(),
        }


# ==================== 全局单例 ====================

_global_service: Optional[PerformanceMonitoringService] = None


def get_monitoring_service(use_memory: bool = False) -> PerformanceMonitoringService:
    """获取性能监控服务实例
    
    Args:
        use_memory: 是否使用内存存储
        
    Returns:
        PerformanceMonitoringService 实例
    """
    global _global_service
    
    if _global_service is None:
        _global_service = PerformanceMonitoringService(use_memory=use_memory)
    
    return _global_service


def reset_monitoring_service():
    """重置全局服务实例（用于测试）"""
    global _global_service
    _global_service = None


# ==================== 导出 ====================

__all__ = [
    'PerformanceMonitoringService',
    'get_monitoring_service',
    'reset_monitoring_service',
]

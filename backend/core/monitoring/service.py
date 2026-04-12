"""统一监控服务

整合系统性能监控、资源指标收集、告警管理等功能。
"""

import asyncio
import logging
import threading
import time
from collections import deque
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any

import psutil

# 条件导入GPUtil
try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    GPUtil = None
    logging.warning("GPUtil not available, GPU monitoring disabled")

from .models import (
    SystemMetrics, GPUMetrics, TrainingMetrics, 
    AlertRule, Alert, MetricType, AlertLevel
)
from .exceptions import MetricsCollectionError, AlertProcessingError

logger = logging.getLogger(__name__)


class UnifiedMonitoringService:
    """统一监控服务"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初始化统一监控服务
        
        Args:
            config: 监控配置
        """
        self.config = config or {}
        
        # 历史数据存储
        self.history_size = self.config.get('history_size', 1000)
        self.system_history = deque(maxlen=self.history_size)
        self.gpu_history = deque(maxlen=self.history_size)
        self.training_history = deque(maxlen=self.history_size)
        
        # 告警规则和活跃告警
        self.alert_rules: List[AlertRule] = []
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history = deque(maxlen=1000)
        
        # 监控状态
        self.is_monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.collection_interval = self.config.get('collection_interval', 10.0)
        
        # 回调函数
        self.alert_callbacks: List[Callable[[Alert], None]] = []
        
        # 初始化默认告警规则
        self._init_default_alert_rules()
        
    def _init_default_alert_rules(self):
        """初始化默认告警规则"""
        default_rules = [
            AlertRule(
                id="high_cpu_usage",
                name="CPU使用率过高",
                description="CPU使用率超过阈值",
                metric_type=MetricType.CPU,
                metric_name="cpu_percent",
                threshold=90.0,
                operator=">",
                duration=60,
                severity=AlertLevel.HIGH
            ),
            AlertRule(
                id="high_memory_usage",
                name="内存使用率过高",
                description="内存使用率超过阈值",
                metric_type=MetricType.MEMORY,
                metric_name="memory_percent",
                threshold=85.0,
                operator=">",
                duration=30,
                severity=AlertLevel.MEDIUM
            ),
            AlertRule(
                id="low_disk_space",
                name="磁盘空间不足",
                description="磁盘使用率超过阈值",
                metric_type=MetricType.DISK,
                metric_name="disk_percent",
                threshold=90.0,
                operator=">",
                duration=300,
                severity=AlertLevel.HIGH
            ),
            AlertRule(
                id="high_gpu_utilization",
                name="GPU利用率过高",
                description="GPU利用率超过阈值",
                metric_type=MetricType.GPU,
                metric_name="gpu_utilization",
                threshold=95.0,
                operator=">",
                duration=120,
                severity=AlertLevel.MEDIUM
            ),
            AlertRule(
                id="high_gpu_temperature",
                name="GPU温度过高",
                description="GPU温度超过阈值",
                metric_type=MetricType.GPU,
                metric_name="temperature",
                threshold=85.0,
                operator=">",
                duration=30,
                severity=AlertLevel.HIGH
            ),
            AlertRule(
                id="training_loss_spike",
                name="训练损失激增",
                description="训练损失超过阈值",
                metric_type=MetricType.TRAINING,
                metric_name="loss",
                threshold=10.0,
                operator=">",
                duration=0,
                severity=AlertLevel.MEDIUM
            ),
            AlertRule(
                id="low_training_speed",
                name="训练速度过慢",
                description="训练速度低于阈值",
                metric_type=MetricType.TRAINING,
                metric_name="samples_per_second",
                threshold=1.0,
                operator="<",
                duration=60,
                severity=AlertLevel.LOW
            ),
        ]
        self.alert_rules.extend(default_rules)
        
    def start_monitoring(self):
        """开始监控"""
        if self.is_monitoring:
            logger.warning("监控已在运行中")
            return
            
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("统一监控服务已启动")
        
    def stop_monitoring(self):
        """停止监控"""
        if not self.is_monitoring:
            return
            
        logger.info("正在停止统一监控服务...")
        
        # 设置停止标志
        self.is_monitoring = False
        
        # 等待监控线程结束
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5)
            
        logger.info("统一监控服务已停止")
        
    def _monitoring_loop(self):
        """监控主循环"""
        while self.is_monitoring:
            try:
                start_time = time.time()
                
                # 收集系统指标
                self._collect_system_metrics()
                
                # 收集GPU指标
                if self.config.get('enable_gpu_monitoring', True) and GPU_AVAILABLE:
                    self._collect_gpu_metrics()
                
                # 检查告警
                self._check_alerts()
                
                # 控制监控频率
                elapsed = time.time() - start_time
                sleep_time = max(0, self.collection_interval - elapsed)
                if sleep_time > 0 and self.is_monitoring:
                    time.sleep(sleep_time)
                    
            except Exception as e:
                logger.error(f"监控循环异常: {e}")
                if self.is_monitoring:
                    time.sleep(self.collection_interval)
                else:
                    break
                
    def _collect_system_metrics(self) -> Optional[SystemMetrics]:
        """收集系统性能指标"""
        try:
            # CPU和内存
            cpu_percent = psutil.cpu_percent(interval=None)
            memory = psutil.virtual_memory()
            
            # 磁盘
            disk = psutil.disk_usage('/')
            disk_io = psutil.disk_io_counters()
            
            # 网络
            network = psutil.net_io_counters()
            
            # 负载和进程
            load_avg = psutil.getloadavg() if hasattr(psutil, 'getloadavg') else [0, 0, 0]
            process_count = len(psutil.pids())
            
            metrics = SystemMetrics(
                timestamp=time.time(),
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_used_gb=memory.used / (1024**3),
                memory_total_gb=memory.total / (1024**3),
                disk_percent=disk.percent,
                disk_io_read_mb=disk_io.read_bytes / (1024**2) if disk_io else 0,
                disk_io_write_mb=disk_io.write_bytes / (1024**2) if disk_io else 0,
                network_sent_mb=network.bytes_sent / (1024**2),
                network_recv_mb=network.bytes_recv / (1024**2),
                load_average=list(load_avg),
                process_count=process_count
            )
            
            self.system_history.append(metrics)
            return metrics
            
        except Exception as e:
            logger.error(f"系统指标收集失败: {e}")
            raise MetricsCollectionError(f"系统指标收集失败: {e}")
            
    def _collect_gpu_metrics(self) -> List[GPUMetrics]:
        """收集GPU性能指标"""
        gpu_metrics = []
        
        try:
            if not GPU_AVAILABLE or GPUtil is None:
                return gpu_metrics
                
            gpus = GPUtil.getGPUs()
            for gpu in gpus:
                metrics = GPUMetrics(
                    timestamp=time.time(),
                    gpu_id=gpu.id,
                    gpu_name=gpu.name,
                    gpu_utilization=gpu.load * 100,
                    memory_utilization=gpu.memoryUtil * 100,
                    memory_used_mb=gpu.memoryUsed,
                    memory_total_mb=gpu.memoryTotal,
                    temperature=gpu.temperature
                )
                
                gpu_metrics.append(metrics)
                
            # 添加到历史记录
            for metrics in gpu_metrics:
                self.gpu_history.append(metrics)
                
        except Exception as e:
            logger.error(f"GPU指标收集失败: {e}")
            raise MetricsCollectionError(f"GPU指标收集失败: {e}")
            
        return gpu_metrics
        
    def add_training_metrics(self, metrics: TrainingMetrics):
        """添加训练指标"""
        self.training_history.append(metrics)
        
    def _check_alerts(self):
        """检查告警条件"""
        current_time = time.time()
        
        for rule in self.alert_rules:
            if not rule.enabled:
                continue
                
            try:
                # 获取最新指标
                latest_metrics = self._get_latest_metrics(rule.metric_type)
                if not latest_metrics:
                    continue
                    
                # 检查阈值
                metric_value = getattr(latest_metrics, rule.metric_name, None)
                if metric_value is None:
                    continue
                    
                # 评估条件
                condition_met = self._evaluate_condition(
                    metric_value, rule.threshold, rule.operator
                )
                
                alert_id = f"{rule.id}_{rule.metric_type.value}"
                
                if condition_met:
                    # 检查是否需要触发新告警
                    if alert_id not in self.active_alerts:
                        # 检查持续时间
                        if self._check_duration(rule, current_time):
                            alert = Alert(
                                alert_id=f"{alert_id}_{int(current_time)}",
                                rule_id=rule.id,
                                name=rule.name,
                                description=rule.description,
                                level=rule.severity,
                                timestamp=datetime.fromtimestamp(current_time),
                                metric_value=metric_value,
                                threshold=rule.threshold
                            )
                            
                            self.active_alerts[alert_id] = alert
                            self.alert_history.append(alert)
                            
                            # 触发回调
                            for callback in self.alert_callbacks:
                                try:
                                    callback(alert)
                                except Exception as e:
                                    logger.error(f"告警回调执行失败: {e}")
                                    
                            logger.warning(f"触发告警: {alert.name} - {alert.description}")
                else:
                    # 条件不满足，解除告警
                    if alert_id in self.active_alerts:
                        alert = self.active_alerts[alert_id]
                        alert.resolved = True
                        alert.resolved_at = datetime.fromtimestamp(current_time)
                        del self.active_alerts[alert_id]
                        
                        logger.info(f"告警已解除: {alert.name} - {alert.description}")
                        
            except Exception as e:
                logger.error(f"告警检查失败 {rule.name}: {e}")
                raise AlertProcessingError(f"告警检查失败 {rule.name}: {e}")
                
    def _get_latest_metrics(self, metric_type: MetricType):
        """获取最新指标"""
        if metric_type == MetricType.SYSTEM and self.system_history:
            return self.system_history[-1]
        elif metric_type == MetricType.GPU and self.gpu_history:
            return self.gpu_history[-1]
        elif metric_type == MetricType.TRAINING and self.training_history:
            return self.training_history[-1]
        return None
        
    def _evaluate_condition(self, value: float, threshold: float, operator: str) -> bool:
        """评估告警条件"""
        if operator == ">":
            return value > threshold
        elif operator == "<":
            return value < threshold
        elif operator == ">=":
            return value >= threshold
        elif operator == "<=":
            return value <= threshold
        elif operator == "==":
            return abs(value - threshold) < 1e-6
        elif operator == "!=":
            return abs(value - threshold) >= 1e-6
        return False
        
    def _check_duration(self, rule: AlertRule, current_time: float) -> bool:
        """检查告警持续时间"""
        if rule.duration <= 0:
            return True
            
        # 检查历史数据中是否持续满足条件
        start_time = current_time - rule.duration
        
        if rule.metric_type == MetricType.SYSTEM:
            history = self.system_history
        elif rule.metric_type == MetricType.GPU:
            history = self.gpu_history
        elif rule.metric_type == MetricType.TRAINING:
            history = self.training_history
        else:
            return False
            
        # 检查时间窗口内的所有数据点
        violation_count = 0
        total_count = 0
        
        for metrics in reversed(history):
            if metrics.timestamp < start_time:
                break
                
            total_count += 1
            metric_value = getattr(metrics, rule.metric_name, None)
            if metric_value is not None:
                if self._evaluate_condition(metric_value, rule.threshold, rule.operator):
                    violation_count += 1
                    
        # 要求至少80%的数据点违反阈值
        return total_count > 0 and (violation_count / total_count) >= 0.8
        
    def get_current_metrics(self) -> Dict[str, Any]:
        """获取当前性能指标"""
        result = {
            "timestamp": time.time(),
            "system": None,
            "gpu": [],
            "training": None,
            "alerts": len(self.active_alerts)
        }
        
        if self.system_history:
            result["system"] = self.system_history[-1]
            
        if self.gpu_history:
            # 获取最新的GPU指标
            latest_gpu_time = max(m.timestamp for m in self.gpu_history)
            latest_gpu_metrics = [m for m in self.gpu_history if m.timestamp == latest_gpu_time]
            result["gpu"] = latest_gpu_metrics
            
        if self.training_history:
            result["training"] = self.training_history[-1]
            
        return result
        
    def get_historical_metrics(self, 
                             metric_type: str, 
                             start_time: Optional[float] = None,
                             end_time: Optional[float] = None) -> List[Any]:
        """获取历史性能指标"""
        if metric_type == "system":
            history = self.system_history
        elif metric_type == "gpu":
            history = self.gpu_history
        elif metric_type == "training":
            history = self.training_history
        else:
            return []
            
        # 时间过滤
        filtered_metrics = []
        for metrics in history:
            if start_time and metrics.timestamp < start_time:
                continue
            if end_time and metrics.timestamp > end_time:
                continue
            filtered_metrics.append(metrics)
            
        return filtered_metrics
        
    def get_active_alerts(self) -> List[Alert]:
        """获取活跃告警"""
        return list(self.active_alerts.values())
        
    def add_alert_rule(self, rule: AlertRule):
        """添加告警规则"""
        self.alert_rules.append(rule)
        logger.info(f"添加告警规则: {rule.name}")
        
    def remove_alert_rule(self, rule_id: str):
        """移除告警规则"""
        self.alert_rules = [r for r in self.alert_rules if r.id != rule_id]
        logger.info(f"移除告警规则: {rule_id}")
        
    def update_alert_rule(self, rule_id: str, **kwargs):
        """更新告警规则"""
        for rule in self.alert_rules:
            if rule.id == rule_id:
                for key, value in kwargs.items():
                    if hasattr(rule, key):
                        setattr(rule, key, value)
                logger.info(f"更新告警规则: {rule_id}")
                break
                
    def add_alert_callback(self, callback: Callable[[Alert], None]):
        """添加告警回调函数"""
        self.alert_callbacks.append(callback)
    
    def get_status(self) -> Dict[str, Any]:
        """获取监控器状态"""
        return {
            'status': 'running' if self.is_monitoring else 'stopped',
            'active_alerts': len(self.active_alerts),
            'system_metrics_count': len(self.system_history),
            'gpu_metrics_count': len(self.gpu_history),
            'training_metrics_count': len(self.training_history)
        }
    
    def get_metrics_history(self, start_time: Optional[float] = None, end_time: Optional[float] = None, interval: str = '5m') -> Dict[str, List]:
        """获取指标历史数据"""
        try:
            # 过滤系统指标
            system_metrics = []
            for metrics in self.system_history:
                if start_time and metrics.timestamp < start_time:
                    continue
                if end_time and metrics.timestamp > end_time:
                    continue
                system_metrics.append({
                    'timestamp': metrics.timestamp,
                    'cpu_percent': metrics.cpu_percent,
                    'memory_percent': metrics.memory_percent,
                    'disk_percent': metrics.disk_percent
                })
            
            # 过滤GPU指标
            gpu_metrics = []
            for metrics in self.gpu_history:
                if start_time and metrics.timestamp < start_time:
                    continue
                if end_time and metrics.timestamp > end_time:
                    continue
                gpu_metrics.append({
                    'timestamp': metrics.timestamp,
                    'gpu_id': metrics.gpu_id,
                    'gpu_utilization': metrics.gpu_utilization,
                    'temperature': metrics.temperature
                })
            
            # 过滤训练指标
            training_metrics = []
            for metrics in self.training_history:
                if start_time and metrics.timestamp < start_time:
                    continue
                if end_time and metrics.timestamp > end_time:
                    continue
                training_metrics.append({
                    'timestamp': metrics.timestamp,
                    'session_id': metrics.session_id,
                    'loss': metrics.loss,
                    'accuracy': metrics.accuracy
                })
            
            return {
                'cpu_usage': system_metrics,
                'gpu_usage': gpu_metrics,
                'training_metrics': training_metrics
            }
        except Exception as e:
            logger.error(f"获取指标历史数据失败: {e}")
            return {}
    
    def get_training_metrics(self, session_id: str, metric_types: Optional[List[str]] = None, start_time: Optional[float] = None, end_time: Optional[float] = None) -> Dict[str, List]:
        """获取训练指标"""
        try:
            metrics_data = {}
            
            # 过滤训练指标
            filtered_metrics = []
            for metrics in self.training_history:
                if metrics.session_id != session_id:
                    continue
                if start_time and metrics.timestamp < start_time:
                    continue
                if end_time and metrics.timestamp > end_time:
                    continue
                filtered_metrics.append(metrics)
            
            # 如果没有指定指标类型，默认返回所有类型
            if not metric_types:
                metric_types = ['loss', 'accuracy', 'learning_rate', 'samples_per_second']
            
            # 按类型组织数据
            for metric_type in metric_types:
                metrics_data[metric_type] = []
                for metrics in filtered_metrics:
                    if hasattr(metrics, metric_type) and getattr(metrics, metric_type) is not None:
                        metrics_data[metric_type].append({
                            'timestamp': metrics.timestamp,
                            'value': getattr(metrics, metric_type),
                            'epoch': getattr(metrics, 'epoch', None),
                            'step': getattr(metrics, 'step', None)
                        })
            
            return metrics_data
        except Exception as e:
            logger.error(f"获取训练指标失败: {e}")
            return {}
    
    def get_training_logs(self, session_id: str, level: Optional[str] = None, limit: int = 100, offset: int = 0, start_time: Optional[float] = None, end_time: Optional[float] = None) -> List[Dict]:
        """获取训练日志"""
        # 注意：在当前实现中，训练日志存储在训练历史中
        # 这里简化实现，返回训练指标作为日志
        try:
            logs = []
            for metrics in self.training_history:
                if metrics.session_id != session_id:
                    continue
                if start_time and metrics.timestamp < start_time:
                    continue
                if end_time and metrics.timestamp > end_time:
                    continue
                
                log_entry = {
                    'timestamp': metrics.timestamp,
                    'level': 'INFO',
                    'message': f"Training metrics - Loss: {metrics.loss}, Accuracy: {metrics.accuracy}",
                    'session_id': metrics.session_id
                }
                logs.append(log_entry)
            
            # 应用偏移和限制
            logs = logs[offset:offset+limit]
            return logs
        except Exception as e:
            logger.error(f"获取训练日志失败: {e}")
            return []
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """获取性能摘要"""
        current_metrics = self.get_current_metrics()
        
        return {
            'timestamp': time.time(),
            'current_metrics': current_metrics,
            'active_alerts_count': len(self.active_alerts),
            'system_metrics_count': len(self.system_history),
            'gpu_metrics_count': len(self.gpu_history),
            'training_metrics_count': len(self.training_history),
            'status': 'monitoring' if self.is_monitoring else 'stopped'
        }

    def get_resource_usage(self, resource_type: Optional[str] = None, time_range: str = '1h') -> Dict[str, Any]:
        """获取资源使用情况"""
        try:
            # 解析时间范围
            hours = 1
            if time_range.endswith('h'):
                hours = int(time_range[:-1])
            elif time_range.endswith('d'):
                hours = int(time_range[:-1]) * 24
            
            end_time = time.time()
            start_time = end_time - (hours * 3600)
            
            usage_data = {}
            
            if not resource_type or resource_type == 'cpu':
                cpu_usage = []
                for metrics in self.system_history:
                    if start_time <= metrics.timestamp <= end_time:
                        cpu_usage.append({
                            'timestamp': metrics.timestamp,
                            'value': metrics.cpu_percent
                        })
                usage_data['cpu'] = cpu_usage
            
            if not resource_type or resource_type == 'memory':
                memory_usage = []
                for metrics in self.system_history:
                    if start_time <= metrics.timestamp <= end_time:
                        memory_usage.append({
                            'timestamp': metrics.timestamp,
                            'value': metrics.memory_percent
                        })
                usage_data['memory'] = memory_usage
            
            if not resource_type or resource_type == 'gpu':
                gpu_usage = []
                for metrics in self.gpu_history:
                    if start_time <= metrics.timestamp <= end_time:
                        gpu_usage.append({
                            'timestamp': metrics.timestamp,
                            'gpu_id': metrics.gpu_id,
                            'value': metrics.gpu_utilization
                        })
                usage_data['gpu'] = gpu_usage
            
            return usage_data
        except Exception as e:
            logger.error(f"获取资源使用情况失败: {e}")
            return {}


# 全局监控服务实例
_global_monitoring_service: Optional[UnifiedMonitoringService] = None


def get_monitoring_service() -> UnifiedMonitoringService:
    """获取全局监控服务实例"""
    global _global_monitoring_service
    if _global_monitoring_service is None:
        _global_monitoring_service = UnifiedMonitoringService()
    return _global_monitoring_service


def start_global_monitoring():
    """启动全局监控"""
    service = get_monitoring_service()
    service.start_monitoring()
    

def stop_global_monitoring():
    """停止全局监控"""
    global _global_monitoring_service
    if _global_monitoring_service:
        _global_monitoring_service.stop_monitoring()
        _global_monitoring_service = None
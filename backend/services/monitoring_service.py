"""监控服务 - 已重构为使用统一核心服务

此文件已重构为使用统一的监控核心服务，避免重复代码。
"""

from backend.core.monitoring.service import (
    UnifiedMonitoringService,
    get_monitoring_service as get_core_monitoring_service
)
from backend.core.monitoring.models import (
    SystemMetrics, GPUMetrics, TrainingMetrics, AlertRule, Alert
)

# 为了向后兼容，重新导出核心服务的类和函数
PerformanceMonitor = UnifiedMonitoringService
get_global_monitor = get_core_monitoring_service

# 导出所有模型类以保持向后兼容
__all__ = [
    'PerformanceMonitor',
    'get_global_monitor',
    'SystemMetrics',
    'GPUMetrics',
    'TrainingMetrics',
    'AlertRule',
    'Alert'
]

logger = logging.getLogger(__name__)


class MonitoringService:
    """监控服务"""
    
    def __init__(self, config=None):
        """初始化监控服务
        
        Args:
            config: 监控配置
        """
        self.config = config or get_monitoring_config()
        
        # 获取各个组件实例
        self.performance_monitor = get_global_monitor()
        self.anomaly_detector = get_global_detector()
        self.alert_manager = get_global_alert_manager()
        self.dashboard_manager = get_global_dashboard_manager()
        
    def start_monitoring(self):
        """启动监控"""
        try:
            # 启动性能监控
            self.performance_monitor.start_monitoring()
            logger.info("性能监控已启动")
            
            # 启动告警管理器
            self.alert_manager.start()
            logger.info("告警管理器已启动")
            
        except Exception as e:
            logger.error(f"启动监控失败: {e}")
            raise
            
    def start(self):
        """启动监控服务（别名）"""
        return self.start_monitoring()
            
    def stop_monitoring(self):
        """停止监控"""
        try:
            # 停止性能监控
            self.performance_monitor.stop_monitoring()
            logger.info("性能监控已停止")
            
            # 停止告警管理器
            self.alert_manager.stop()
            logger.info("告警管理器已停止")
            
        except Exception as e:
            logger.error(f"停止监控失败: {e}")
            raise
            
    def stop(self):
        """停止监控服务（别名）"""
        return self.stop_monitoring()
            
    def get_current_metrics(self) -> Dict[str, Any]:
        """获取当前指标"""
        try:
            return self.performance_monitor.get_current_metrics()
        except Exception as e:
            logger.error(f"获取当前指标失败: {e}")
            raise
            
    def get_historical_metrics(self, 
                             metric_type: str,
                             start_time: Optional[float] = None,
                             end_time: Optional[float] = None) -> List[Any]:
        """获取历史指标"""
        try:
            return self.performance_monitor.get_historical_metrics(
                metric_type, start_time, end_time
            )
        except Exception as e:
            logger.error(f"获取历史指标失败: {e}")
            raise
            
    def add_training_metrics(self, metrics: TrainingMetrics):
        """添加训练指标"""
        try:
            self.performance_monitor.add_training_metrics(metrics)
        except Exception as e:
            logger.error(f"添加训练指标失败: {e}")
            raise
            
    def get_active_alerts(self) -> List[Alert]:
        """获取活跃告警"""
        try:
            return self.alert_manager.get_active_alerts()
        except Exception as e:
            logger.error(f"获取活跃告警失败: {e}")
            raise
            
    def get_alert_history(self, limit: int = 100) -> List[Alert]:
        """获取告警历史"""
        try:
            return self.alert_manager.get_alert_history(limit)
        except Exception as e:
            logger.error(f"获取告警历史失败: {e}")
            raise
            
    def add_alert_rule(self, rule: AlertRule):
        """添加告警规则"""
        try:
            self.alert_manager.add_alert_rule(rule)
        except Exception as e:
            logger.error(f"添加告警规则失败: {e}")
            raise
            
    def remove_alert_rule(self, rule_id: str):
        """移除告警规则"""
        try:
            self.alert_manager.remove_alert_rule(rule_id)
        except Exception as e:
            logger.error(f"移除告警规则失败: {e}")
            raise
            
    def detect_anomalies(self, 
                        system_data: Optional[List[SystemMetrics]] = None,
                        gpu_data: Optional[List[GPUMetrics]] = None,
                        training_data: Optional[List[TrainingMetrics]] = None) -> Dict[str, List[AnomalyResult]]:
        """检测异常"""
        try:
            return self.anomaly_detector.detect_all(
                system_data, gpu_data, training_data
            )
        except Exception as e:
            logger.error(f"异常检测失败: {e}")
            raise
            
    def get_anomaly_summary(self, hours: int = 24) -> Dict[str, Any]:
        """获取异常摘要"""
        try:
            return self.anomaly_detector.get_comprehensive_summary(hours)
        except Exception as e:
            logger.error(f"获取异常摘要失败: {e}")
            raise
            
    def generate_dashboard_data(self, layout_id: str) -> Dict[str, Any]:
        """生成仪表板数据"""
        try:
            return self.dashboard_manager.generate_dashboard_data(layout_id)
        except Exception as e:
            logger.error(f"生成仪表板数据失败: {e}")
            raise
            
    def get_dashboard_statistics(self) -> Dict[str, Any]:
        """获取仪表板统计信息"""
        try:
            return self.dashboard_manager.get_dashboard_statistics()
        except Exception as e:
            logger.error(f"获取仪表板统计失败: {e}")
            raise
            
    def get_system_statistics(self) -> Dict[str, Any]:
        """获取系统统计信息"""
        try:
            # 获取各组件统计
            monitor_stats = self.performance_monitor.get_current_metrics()
            alert_stats = self.alert_manager.get_statistics()
            dashboard_stats = self.dashboard_manager.get_dashboard_statistics()
            anomaly_stats = self.anomaly_detector.get_comprehensive_summary()
            
            return {
                'monitor': monitor_stats,
                'alerts': alert_stats,
                'dashboards': dashboard_stats,
                'anomalies': anomaly_stats,
                'timestamp': datetime.now().timestamp()
            }
            
        except Exception as e:
            logger.error(f"获取系统统计失败: {e}")
            raise
    
    def get_system_health(self) -> Dict[str, Any]:
        """获取系统健康状态"""
        try:
            # 获取性能监控状态
            monitor_status = self.performance_monitor.get_status()
            
            # 获取告警管理器状态
            alert_status = self.alert_manager.get_status()
            
            # 获取数据库连接状态
            from backend.modules.database import get_database_manager
            db_manager = get_database_manager()
            db_health = db_manager.health_check()
            
            # 获取Redis连接状态
            from backend.core.redis_client import get_redis_client
            redis_client = get_redis_client()
            try:
                redis_client.ping()
                redis_health = True
            except:
                redis_health = False
            
            return {
                'status': 'healthy' if db_health and redis_health else 'unhealthy',
                'components': {
                    'database': {
                        'status': 'healthy' if db_health else 'unhealthy',
                        'response_time': monitor_status.get('database_response_time', 0)
                    },
                    'redis': {
                        'status': 'healthy' if redis_health else 'unhealthy',
                        'response_time': monitor_status.get('redis_response_time', 0)
                    },
                    'monitoring': {
                        'status': monitor_status.get('status', 'unknown'),
                        'response_time': monitor_status.get('response_time', 0)
                    },
                    'alerts': {
                        'status': alert_status.get('status', 'unknown'),
                        'active_alerts': alert_status.get('active_count', 0)
                    }
                },
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"获取系统健康状态失败: {e}")
            raise
    
    def get_system_metrics(self, start_time: Optional[datetime] = None, end_time: Optional[datetime] = None, interval: str = '5m') -> Dict[str, Any]:
        """获取系统指标"""
        try:
            # 如果没有指定时间范围，默认获取最近1小时的数据
            if end_time is None:
                end_time = datetime.utcnow()
            if start_time is None:
                start_time = end_time - timedelta(hours=1)
            
            # 获取性能指标
            metrics = self.performance_monitor.get_metrics_history(
                start_time=start_time.timestamp(),
                end_time=end_time.timestamp(),
                interval=interval
            )
            
            # 计算摘要信息
            summary = {}
            if 'cpu_usage' in metrics:
                cpu_values = [m['value'] for m in metrics['cpu_usage'] if 'value' in m]
                if cpu_values:
                    summary['avg_cpu'] = sum(cpu_values) / len(cpu_values)
                    summary['peak_cpu'] = max(cpu_values)
            
            if 'memory_usage' in metrics:
                memory_values = [m['value'] for m in metrics['memory_usage'] if 'value' in m]
                if memory_values:
                    summary['avg_memory'] = sum(memory_values) / len(memory_values)
                    summary['peak_memory'] = max(memory_values)
            
            return {
                'metrics': metrics,
                'summary': summary,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"获取系统指标失败: {e}")
            raise
    
    def get_training_metrics(self, session_id: str, metric_types: Optional[List[str]] = None, start_time: Optional[datetime] = None, end_time: Optional[datetime] = None) -> Dict[str, Any]:
        """获取训练会话指标"""
        try:
            # 获取训练指标
            training_metrics = self.performance_monitor.get_training_metrics(
                session_id=session_id,
                metric_types=metric_types,
                start_time=start_time.timestamp() if start_time else None,
                end_time=end_time.timestamp() if end_time else None
            )
            
            # 计算摘要信息
            summary = {}
            if 'accuracy' in training_metrics:
                accuracy_values = [m['value'] for m in training_metrics['accuracy'] if 'value' in m]
                if accuracy_values:
                    summary['best_accuracy'] = max(accuracy_values)
                    summary['final_accuracy'] = accuracy_values[-1] if accuracy_values else 0
            
            if 'loss' in training_metrics:
                loss_values = [m['value'] for m in training_metrics['loss'] if 'value' in m]
                if loss_values:
                    summary['final_loss'] = loss_values[-1] if loss_values else 0
            
            return {
                'metrics': training_metrics,
                'summary': summary,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"获取训练指标失败: {e}")
            raise
    
    def get_training_logs(self, session_id: str, level: Optional[str] = None, limit: int = 100, offset: int = 0, start_time: Optional[datetime] = None, end_time: Optional[datetime] = None) -> Dict[str, Any]:
        """获取训练日志"""
        try:
            # 获取训练日志
            logs = self.performance_monitor.get_training_logs(
                session_id=session_id,
                level=level,
                limit=limit,
                offset=offset,
                start_time=start_time.timestamp() if start_time else None,
                end_time=end_time.timestamp() if end_time else None
            )
            
            return {
                'logs': logs,
                'total': len(logs),
                'has_more': len(logs) >= limit,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"获取训练日志失败: {e}")
            raise
    
    def get_resource_usage(self, resource_type: Optional[str] = None, time_range: str = '1h') -> Dict[str, Any]:
        """获取资源使用情况"""
        try:
            # 获取资源使用情况
            usage = self.performance_monitor.get_resource_usage(
                resource_type=resource_type,
                time_range=time_range
            )
            
            return {
                'usage': usage,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"获取资源使用情况失败: {e}")
            raise
    
    def get_alerts(self, status: str = 'all', severity: Optional[str] = None, limit: int = 50) -> Dict[str, Any]:
        """获取告警信息"""
        try:
            # 获取告警
            alerts = self.alert_manager.get_alerts(
                status=status,
                severity=severity,
                limit=limit
            )
            
            # 计算摘要信息
            total = len(alerts)
            active_count = len([a for a in alerts if a.get('status') == 'active'])
            critical_count = len([a for a in alerts if a.get('severity') == 'critical'])
            
            summary = {
                'total': total,
                'active': active_count,
                'critical': critical_count
            }
            
            return {
                'alerts': alerts,
                'summary': summary,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"获取告警信息失败: {e}")
            raise
    
    def resolve_alert(self, alert_id: str, resolution_note: str = '') -> bool:
        """解决告警"""
        try:
            # 解决告警
            success = self.alert_manager.resolve_alert(
                alert_id=alert_id,
                resolution_note=resolution_note
            )
            
            return success
            
        except Exception as e:
            logger.error(f"解决告警失败: {e}")
            raise
    
    def get_dashboard_summary(self) -> Dict[str, Any]:
        """获取监控仪表板摘要"""
        try:
            # 获取仪表板摘要
            summary = self.dashboard_manager.get_dashboard_summary()
            
            return {
                'summary': summary,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"获取仪表板摘要失败: {e}")
            raise


# 全局监控服务实例
_global_monitoring_service: Optional[MonitoringService] = None


def get_monitoring_service() -> MonitoringService:
    """获取全局监控服务实例"""
    global _global_monitoring_service
    if _global_monitoring_service is None:
        _global_monitoring_service = MonitoringService()
    return _global_monitoring_service


def initialize_monitoring_system(config=None) -> Dict[str, Any]:
    """初始化监控系统
    
    Args:
        config: 监控配置
        
    Returns:
        初始化结果
    """
    try:
        service = MonitoringService(config)
        service.start_monitoring()
        set_monitoring_service(service)
        
        return {
            'success': True,
            'message': '监控系统初始化成功',
            'timestamp': datetime.now().timestamp()
        }
        
    except Exception as e:
        logger.error(f"监控系统初始化失败: {e}")
        return {
            'success': False,
            'message': f'监控系统初始化失败: {str(e)}',
            'error': str(e),
            'timestamp': datetime.now().timestamp()
        }


def shutdown_monitoring_system():
    """关闭监控系统"""
    try:
        global _global_monitoring_service
        if _global_monitoring_service:
            _global_monitoring_service.stop_monitoring()
            _global_monitoring_service = None
            
        # 停止全局实例
        from .manager import stop_global_monitoring
        stop_global_monitoring()
        
        return {
            'success': True,
            'message': '监控系统已关闭',
            'timestamp': datetime.now().timestamp()
        }
        
    except Exception as e:
        logger.error(f"监控系统关闭失败: {e}")
        return {
            'success': False,
            'message': f'监控系统关闭失败: {str(e)}',
            'error': str(e),
            'timestamp': datetime.now().timestamp()
        }


def set_monitoring_service(service: MonitoringService):
    """设置全局监控服务实例"""
    global _global_monitoring_service
    _global_monitoring_service = service


def get_system_status() -> Dict[str, Any]:
    """获取系统状态"""
    try:
        service = get_monitoring_service()
        return service.get_system_statistics()
        
    except Exception as e:
        logger.error(f"获取系统状态失败: {e}")
        return {
            'error': f'获取系统状态失败: {str(e)}',
            'timestamp': datetime.now().timestamp()
        }
"""仪表盘业务逻辑服务

聚合训练、模型、系统资源等多数据源，提供统一的仪表盘数据服务。

此服务层负责：
1. 聚合多个数据仓库的查询结果
2. 实时获取系统资源信息
3. 数据格式转换和业务逻辑处理
4. 缓存管理（可选）

Example:
    >>> service = get_dashboard_service()
    >>> overview = service.get_dashboard_overview(user_id="user123")
    >>> print(overview.to_dict())
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

import psutil

from backend.repositories.dashboard_repository import (
    DashboardRepository,
    get_dashboard_repository
)
from backend.schemas.dashboard_models import (
    DashboardOverview,
    TrainingOverview,
    TrainingTrend,
    TrainingDetailedStats,
    ModelOverview,
    SystemResourceSnapshot,
    GPUResourceSnapshot,
    SystemMetricsHistory,
    UserActivitySummary,
    DashboardFilter,
    DashboardTimeRange,
    MetricGranularity
)

logger = logging.getLogger(__name__)


class DashboardService:
    """仪表盘业务服务
    
    聚合训练、模型、数据集、系统资源等多维度数据，
    为前端仪表盘提供统一的数据访问接口。
    
    Attributes:
        repository: 仪表盘数据仓库
        _metrics_cache: 指标缓存（可选）
        _cache_ttl: 缓存过期时间（秒）
    
    Example:
        >>> service = DashboardService()
        >>> overview = service.get_dashboard_overview(user_id="user123")
    """
    
    def __init__(self, repository: Optional[DashboardRepository] = None):
        """初始化仪表盘服务
        
        Args:
            repository: 数据仓库实例，为 None 时自动获取单例
        """
        self.repository = repository or get_dashboard_repository()
        self._metrics_cache: Dict[str, Any] = {}
        self._cache_ttl = 60  # 缓存60秒
        self._last_cache_time: Optional[datetime] = None
    
    # =========================================================================
    # 仪表盘概览
    # =========================================================================
    
    def get_dashboard_overview(
        self,
        user_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        time_range: DashboardTimeRange = DashboardTimeRange.LAST_24_HOURS
    ) -> DashboardOverview:
        """获取仪表盘概览数据
        
        聚合训练、模型、系统资源等多维度数据，提供统一的概览视图。
        
        Args:
            user_id: 用户ID，用于筛选该用户的数据
            tenant_id: 租户ID，用于多租户隔离
            time_range: 时间范围枚举
            
        Returns:
            DashboardOverview: 仪表盘概览数据对象，包含：
                - training: 训练概览（活跃/完成/失败任务数等）
                - models: 模型概览（总数/已部署/准确率等）
                - system: 系统资源快照（CPU/内存/磁盘使用率）
                - gpu: GPU资源列表（如有GPU）
                - user_activity: 用户活动概要
                - alerts_count: 活跃告警数量
                
        Example:
            >>> service = DashboardService()
            >>> overview = service.get_dashboard_overview(user_id="user123")
            >>> print(f"活跃训练: {overview.training.active_count}")
            >>> print(f"CPU使用率: {overview.system.cpu_usage}%")
        """
        try:
            # 创建过滤器计算时间范围
            filter_obj = DashboardFilter(
                user_id=user_id,
                tenant_id=tenant_id,
                time_range=time_range
            )
            start_date, end_date = filter_obj.get_date_range()
            
            # 获取训练概览
            training_data = self.repository.get_training_overview(
                user_id=user_id,
                tenant_id=tenant_id,
                start_date=start_date,
                end_date=end_date
            )
            training_overview = TrainingOverview(
                active_count=training_data.get('active_count', 0),
                completed_count=training_data.get('completed_count', 0),
                failed_count=training_data.get('failed_count', 0),
                pending_count=training_data.get('pending_count', 0),
                paused_count=training_data.get('paused_count', 0),
                success_rate=training_data.get('success_rate', 0.0),
                total_training_time_hours=training_data.get('total_training_time_hours', 0.0),
                avg_training_time_hours=training_data.get('avg_training_time_hours', 0.0)
            )
            
            # 获取模型概览
            model_data = self.repository.get_model_overview(
                user_id=user_id,
                tenant_id=tenant_id
            )
            model_overview = ModelOverview(
                total_count=model_data.get('total_count', 0),
                deployed_count=model_data.get('deployed_count', 0),
                draft_count=model_data.get('draft_count', 0),
                archived_count=model_data.get('archived_count', 0),
                avg_accuracy=model_data.get('avg_accuracy', 0.0),
                best_accuracy=model_data.get('best_accuracy', 0.0),
                avg_f1_score=model_data.get('avg_f1_score', 0.0),
                total_size_gb=model_data.get('total_size_gb', 0.0)
            )
            
            # 获取系统资源
            system_snapshot = self._get_current_system_resources()
            
            # 获取GPU资源
            gpu_snapshots = self._get_gpu_resources()
            
            # 获取用户活动概要（如果指定了user_id）
            user_activity = None
            if user_id:
                activity_data = self.repository.get_user_activity_summary(
                    user_id=user_id,
                    tenant_id=tenant_id
                )
                user_activity = UserActivitySummary(
                    total_training_count=activity_data.get('total_training_count', 0),
                    total_models_created=activity_data.get('total_models_created', 0),
                    total_datasets_used=activity_data.get('total_datasets_used', 0),
                    last_active_at=datetime.fromisoformat(activity_data['last_active_at']) 
                        if activity_data.get('last_active_at') else None,
                    most_used_model_type=activity_data.get('most_used_model_type', ''),
                    avg_training_time_hours=activity_data.get('avg_training_time_hours', 0.0)
                )
            
            # 获取告警数量
            alerts_count = self.repository.get_active_alerts_count(tenant_id=tenant_id)
            
            return DashboardOverview(
                training=training_overview,
                models=model_overview,
                system=system_snapshot,
                gpu=gpu_snapshots,
                user_activity=user_activity,
                alerts_count=alerts_count,
                last_updated=datetime.utcnow()
            )
            
        except Exception as e:
            logger.error(f"Failed to get dashboard overview: {e}")
            # 返回空的概览对象
            return DashboardOverview()
    
    # =========================================================================
    # 训练统计
    # =========================================================================
    
    def get_training_detailed_stats(
        self,
        user_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        time_range: DashboardTimeRange = DashboardTimeRange.LAST_7_DAYS,
        granularity: MetricGranularity = MetricGranularity.DAY
    ) -> TrainingDetailedStats:
        """获取详细训练统计
        
        提供训练任务的详细统计信息，包括概览、趋势、分布等。
        
        Args:
            user_id: 用户ID
            tenant_id: 租户ID  
            time_range: 时间范围
            granularity: 数据粒度（分钟/小时/天/周/月）
            
        Returns:
            TrainingDetailedStats: 详细训练统计，包含：
                - overview: 训练概览
                - trends: 时间趋势数据
                - by_type: 按类型分布
                - by_status: 按状态分布
                - top_models: 最常训练的模型
                - recent_sessions: 最近的训练会话
                
        Example:
            >>> stats = service.get_training_detailed_stats(
            ...     user_id="user123",
            ...     time_range=DashboardTimeRange.LAST_7_DAYS
            ... )
            >>> for trend in stats.trends:
            ...     print(f"{trend.date_label}: {trend.count} 任务")
        """
        try:
            # 创建过滤器
            filter_obj = DashboardFilter(
                user_id=user_id,
                tenant_id=tenant_id,
                time_range=time_range,
                granularity=granularity
            )
            start_date, end_date = filter_obj.get_date_range()
            
            # 获取概览
            overview_data = self.repository.get_training_overview(
                user_id=user_id,
                tenant_id=tenant_id,
                start_date=start_date,
                end_date=end_date
            )
            overview = TrainingOverview(
                active_count=overview_data.get('active_count', 0),
                completed_count=overview_data.get('completed_count', 0),
                failed_count=overview_data.get('failed_count', 0),
                pending_count=overview_data.get('pending_count', 0),
                paused_count=overview_data.get('paused_count', 0),
                success_rate=overview_data.get('success_rate', 0.0),
                total_training_time_hours=overview_data.get('total_training_time_hours', 0.0),
                avg_training_time_hours=overview_data.get('avg_training_time_hours', 0.0)
            )
            
            # 获取趋势
            trends_data = self.repository.get_training_trends(
                user_id=user_id,
                tenant_id=tenant_id,
                start_date=start_date,
                end_date=end_date,
                granularity=granularity.value
            )
            trends = [
                TrainingTrend(
                    timestamp=datetime.fromisoformat(t['timestamp']) if isinstance(t['timestamp'], str) else t['timestamp'],
                    date_label=t.get('date_label', ''),
                    count=t.get('count', 0),
                    success_count=t.get('success_count', 0),
                    failed_count=t.get('failed_count', 0),
                    avg_duration_hours=t.get('avg_duration_hours', 0.0)
                )
                for t in trends_data
            ]
            
            # 获取按类型分布
            by_type = self.repository.get_training_by_type(
                user_id=user_id,
                tenant_id=tenant_id,
                start_date=start_date,
                end_date=end_date
            )
            
            # 按状态分布
            by_status = {
                'running': overview.active_count,
                'completed': overview.completed_count,
                'failed': overview.failed_count,
                'pending': overview.pending_count,
                'paused': overview.paused_count
            }
            
            # 获取最近会话
            recent_sessions = self.repository.get_recent_training_sessions(
                user_id=user_id,
                tenant_id=tenant_id,
                limit=10
            )
            
            return TrainingDetailedStats(
                overview=overview,
                trends=trends,
                by_type=by_type,
                by_status=by_status,
                top_models=[],  # 可以从关联数据获取
                recent_sessions=recent_sessions
            )
            
        except Exception as e:
            logger.error(f"Failed to get training detailed stats: {e}")
            return TrainingDetailedStats()
    
    # =========================================================================
    # 模型统计
    # =========================================================================
    
    def get_model_stats(
        self,
        user_id: Optional[str] = None,
        tenant_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """获取模型统计数据
        
        提供模型的统计信息，包括概览和分布数据。
        
        Args:
            user_id: 用户ID
            tenant_id: 租户ID
            
        Returns:
            Dict: 模型统计数据，包含：
                - overview: 模型概览（总数、部署数、准确率等）
                - distribution: 分布数据（按类型、框架、状态等）
                - top_models: 表现最好的模型列表
                
        Example:
            >>> stats = service.get_model_stats(user_id="user123")
            >>> print(f"总模型数: {stats['overview']['total_count']}")
            >>> print(f"平均准确率: {stats['overview']['avg_accuracy']}")
        """
        try:
            # 获取概览
            overview = self.repository.get_model_overview(
                user_id=user_id,
                tenant_id=tenant_id
            )
            
            # 获取分布
            distribution = self.repository.get_model_distribution(
                user_id=user_id,
                tenant_id=tenant_id
            )
            
            # 获取Top模型
            top_models = self.repository.get_top_models(
                user_id=user_id,
                tenant_id=tenant_id,
                limit=5,
                order_by='accuracy'
            )
            
            return {
                'overview': overview,
                'distribution': distribution,
                'top_models': top_models
            }
            
        except Exception as e:
            logger.error(f"Failed to get model stats: {e}")
            return {
                'overview': {},
                'distribution': {},
                'top_models': []
            }
    
    # =========================================================================
    # 数据集统计
    # =========================================================================
    
    def get_dataset_stats(
        self,
        user_id: Optional[str] = None,
        tenant_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """获取数据集统计
        
        Args:
            user_id: 用户ID
            tenant_id: 租户ID
            
        Returns:
            Dict: 数据集统计数据
                - total_count: 数据集总数
                - active_count: 活跃数据集数
                - archived_count: 归档数据集数
                - processing_count: 处理中数据集数
                
        Example:
            >>> stats = service.get_dataset_stats(user_id="user123")
            >>> print(f"总数据集: {stats['total_count']}")
        """
        try:
            return self.repository.get_dataset_overview(
                user_id=user_id,
                tenant_id=tenant_id
            )
        except Exception as e:
            logger.error(f"Failed to get dataset stats: {e}")
            return {
                'total_count': 0,
                'active_count': 0,
                'archived_count': 0,
                'processing_count': 0
            }
    
    # =========================================================================
    # 系统资源
    # =========================================================================
    
    def _get_current_system_resources(self) -> SystemResourceSnapshot:
        """获取当前系统资源快照
        
        使用 psutil 获取实时系统资源使用情况。
        
        Returns:
            SystemResourceSnapshot: 系统资源快照
        """
        try:
            # CPU
            cpu_usage = psutil.cpu_percent(interval=0.1)
            cpu_count = psutil.cpu_count()
            
            # 内存
            memory = psutil.virtual_memory()
            memory_usage = memory.percent
            memory_used_gb = memory.used / (1024 ** 3)
            memory_total_gb = memory.total / (1024 ** 3)
            
            # 磁盘
            disk = psutil.disk_usage('/')
            disk_usage = disk.percent
            disk_used_gb = disk.used / (1024 ** 3)
            disk_total_gb = disk.total / (1024 ** 3)
            
            # 网络
            net_io = psutil.net_io_counters()
            network_sent_mb = net_io.bytes_sent / (1024 ** 2)
            network_recv_mb = net_io.bytes_recv / (1024 ** 2)
            
            return SystemResourceSnapshot(
                timestamp=datetime.utcnow(),
                cpu_usage=cpu_usage,
                cpu_count=cpu_count,
                memory_usage=memory_usage,
                memory_used_gb=memory_used_gb,
                memory_total_gb=memory_total_gb,
                disk_usage=disk_usage,
                disk_used_gb=disk_used_gb,
                disk_total_gb=disk_total_gb,
                network_sent_mb=network_sent_mb,
                network_recv_mb=network_recv_mb
            )
            
        except Exception as e:
            logger.error(f"Failed to get system resources: {e}")
            return SystemResourceSnapshot()
    
    def _get_gpu_resources(self) -> List[GPUResourceSnapshot]:
        """获取GPU资源快照
        
        尝试使用 nvidia-smi 或 pynvml 获取GPU信息。
        
        Returns:
            List[GPUResourceSnapshot]: GPU资源快照列表
        """
        gpu_list = []
        
        try:
            # 尝试使用 pynvml
            import pynvml
            pynvml.nvmlInit()
            
            device_count = pynvml.nvmlDeviceGetCount()
            
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                name = pynvml.nvmlDeviceGetName(handle)
                if isinstance(name, bytes):
                    name = name.decode('utf-8')
                
                # 获取利用率
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                
                # 获取显存
                memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                memory_used_gb = memory_info.used / (1024 ** 3)
                memory_total_gb = memory_info.total / (1024 ** 3)
                memory_usage = (memory_info.used / memory_info.total) * 100 if memory_info.total > 0 else 0
                
                # 获取温度
                try:
                    temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                except:
                    temperature = 0
                
                # 获取功耗
                try:
                    power_draw = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000  # mW to W
                    power_limit = pynvml.nvmlDeviceGetPowerManagementLimit(handle) / 1000
                except:
                    power_draw = 0
                    power_limit = 0
                
                gpu_list.append(GPUResourceSnapshot(
                    device_id=i,
                    name=name,
                    utilization=utilization.gpu,
                    memory_used_gb=memory_used_gb,
                    memory_total_gb=memory_total_gb,
                    memory_usage=memory_usage,
                    temperature=temperature,
                    power_draw=power_draw,
                    power_limit=power_limit
                ))
            
            pynvml.nvmlShutdown()
            
        except ImportError:
            logger.debug("pynvml not available, GPU monitoring disabled")
        except Exception as e:
            logger.debug(f"Failed to get GPU info: {e}")
        
        return gpu_list
    
    def get_system_metrics_history(
        self,
        hours: int = 24,
        granularity: MetricGranularity = MetricGranularity.HOUR
    ) -> SystemMetricsHistory:
        """获取系统指标历史数据
        
        获取过去一段时间的系统指标历史记录。
        
        Args:
            hours: 获取多少小时的历史数据
            granularity: 数据粒度
            
        Returns:
            SystemMetricsHistory: 系统指标历史数据
            
        Note:
            由于系统指标通常存储在时序数据库中，
            此方法目前返回模拟数据，实际实现需要接入监控系统。
        """
        try:
            # 获取当前快照
            current = self._get_current_system_resources()
            
            # 生成历史数据点（模拟）
            # 实际实现中应该从监控数据库获取
            cpu_usage = []
            memory_usage = []
            disk_usage = []
            network_io = []
            
            now = datetime.utcnow()
            points = min(hours, 168)  # 最多一周
            
            for i in range(points):
                timestamp = now - timedelta(hours=i)
                
                # 基于当前值生成历史数据（加入随机波动）
                import random
                
                cpu_val = max(0, min(100, current.cpu_usage + random.uniform(-15, 15)))
                mem_val = max(0, min(100, current.memory_usage + random.uniform(-10, 10)))
                disk_val = max(0, min(100, current.disk_usage + random.uniform(-2, 2)))
                
                cpu_usage.append({
                    'timestamp': timestamp.isoformat(),
                    'value': round(cpu_val, 2)
                })
                
                memory_usage.append({
                    'timestamp': timestamp.isoformat(),
                    'value': round(mem_val, 2)
                })
                
                disk_usage.append({
                    'timestamp': timestamp.isoformat(),
                    'value': round(disk_val, 2)
                })
                
                network_io.append({
                    'timestamp': timestamp.isoformat(),
                    'sent_mb': round(current.network_sent_mb * (1 + random.uniform(-0.1, 0.1)), 2),
                    'recv_mb': round(current.network_recv_mb * (1 + random.uniform(-0.1, 0.1)), 2)
                })
            
            # 按时间升序排列
            cpu_usage.reverse()
            memory_usage.reverse()
            disk_usage.reverse()
            network_io.reverse()
            
            # GPU历史（如有）
            gpu_usage = []
            gpus = self._get_gpu_resources()
            if gpus:
                for i in range(points):
                    timestamp = now - timedelta(hours=points - i - 1)
                    for gpu in gpus:
                        import random
                        gpu_usage.append({
                            'timestamp': timestamp.isoformat(),
                            'device_id': gpu.device_id,
                            'utilization': round(max(0, min(100, gpu.utilization + random.uniform(-20, 20))), 2),
                            'memory_usage': round(max(0, min(100, gpu.memory_usage + random.uniform(-10, 10))), 2)
                        })
            
            return SystemMetricsHistory(
                cpu_usage=cpu_usage,
                memory_usage=memory_usage,
                disk_usage=disk_usage,
                network_io=network_io,
                gpu_usage=gpu_usage
            )
            
        except Exception as e:
            logger.error(f"Failed to get system metrics history: {e}")
            return SystemMetricsHistory()
    
    def get_current_system_snapshot(self) -> SystemResourceSnapshot:
        """获取当前系统资源快照（公开方法）
        
        Returns:
            SystemResourceSnapshot: 当前系统资源快照
        """
        return self._get_current_system_resources()
    
    def get_current_gpu_snapshot(self) -> List[GPUResourceSnapshot]:
        """获取当前GPU资源快照
        
        Returns:
            List[GPUResourceSnapshot]: GPU资源快照列表
        """
        return self._get_gpu_resources()
    
    # =========================================================================
    # 用户活动
    # =========================================================================
    
    def get_user_activity(
        self,
        user_id: str,
        tenant_id: Optional[str] = None
    ) -> UserActivitySummary:
        """获取用户活动概要
        
        Args:
            user_id: 用户ID
            tenant_id: 租户ID
            
        Returns:
            UserActivitySummary: 用户活动概要
        """
        try:
            activity_data = self.repository.get_user_activity_summary(
                user_id=user_id,
                tenant_id=tenant_id
            )
            
            return UserActivitySummary(
                total_training_count=activity_data.get('total_training_count', 0),
                total_models_created=activity_data.get('total_models_created', 0),
                total_datasets_used=activity_data.get('total_datasets_used', 0),
                last_active_at=datetime.fromisoformat(activity_data['last_active_at']) 
                    if activity_data.get('last_active_at') else None,
                most_used_model_type=activity_data.get('most_used_model_type', ''),
                avg_training_time_hours=activity_data.get('avg_training_time_hours', 0.0)
            )
            
        except Exception as e:
            logger.error(f"Failed to get user activity: {e}")
            return UserActivitySummary()
    
    # =========================================================================
    # 告警
    # =========================================================================
    
    def get_alerts_summary(
        self,
        tenant_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """获取告警概要
        
        Args:
            tenant_id: 租户ID
            
        Returns:
            Dict: 告警概要
                - active_count: 活跃告警数
                - critical_count: 严重告警数
                - warning_count: 警告数
        """
        try:
            active_count = self.repository.get_active_alerts_count(tenant_id=tenant_id)
            
            return {
                'active_count': active_count,
                'critical_count': 0,  # 可以从数据库获取详细分类
                'warning_count': 0
            }
            
        except Exception as e:
            logger.error(f"Failed to get alerts summary: {e}")
            return {
                'active_count': 0,
                'critical_count': 0,
                'warning_count': 0
            }


    # =========================================================================
    # 统计扩展方法
    # =========================================================================
    
    def get_training_progress_trend(
        self,
        user_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        days: int = 7
    ) -> Dict[str, Any]:
        """获取训练进度趋势数据
        
        按日期统计训练任务的完成、运行、失败数量趋势。
        
        Args:
            user_id: 用户ID，用于筛选该用户的数据
            tenant_id: 租户ID
            days: 统计天数，默认7天
            
        Returns:
            Dict: 趋势数据，包含：
                - trend: 每日趋势列表
                - period: 统计周期描述
                - summary: 汇总统计
                
        Example:
            >>> service = get_dashboard_service()
            >>> trend = service.get_training_progress_trend(user_id="user123", days=7)
            >>> print(f"总完成: {trend['summary']['total_completed']}")
        """
        try:
            trend_data = self.repository.get_training_progress_trend(
                user_id=user_id,
                tenant_id=tenant_id,
                days=days
            )
            
            # 计算汇总
            total_completed = sum(d['completed'] for d in trend_data)
            total_failed = sum(d['failed'] for d in trend_data)
            total_running = sum(d['running'] for d in trend_data)
            
            return {
                'trend': trend_data,
                'period': f'{days} days',
                'summary': {
                    'total_completed': total_completed,
                    'total_failed': total_failed,
                    'total_running': total_running,
                    'avg_daily_completed': round(total_completed / days, 2) if days > 0 else 0,
                    'success_rate': round(total_completed / (total_completed + total_failed), 4) if (total_completed + total_failed) > 0 else 0
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get training progress trend: {e}")
            return {
                'trend': [],
                'period': f'{days} days',
                'summary': {
                    'total_completed': 0,
                    'total_failed': 0,
                    'total_running': 0,
                    'avg_daily_completed': 0,
                    'success_rate': 0
                }
            }
    
    def get_active_training_tasks(
        self,
        user_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        limit: int = 20
    ) -> Dict[str, Any]:
        """获取活跃训练任务列表
        
        获取正在运行、等待中或暂停的训练任务。
        
        Args:
            user_id: 用户ID
            tenant_id: 租户ID
            limit: 返回数量限制
            
        Returns:
            Dict: 活跃任务数据，包含：
                - active_tasks: 活跃任务列表
                - total_count: 总数量
                - running_count: 运行中数量
                - pending_count: 等待中数量
                
        Example:
            >>> tasks = service.get_active_training_tasks(user_id="user123")
            >>> for task in tasks['active_tasks']:
            ...     print(f"{task['name']}: {task['progress']}%")
        """
        try:
            tasks = self.repository.get_active_training_sessions(
                user_id=user_id,
                tenant_id=tenant_id,
                limit=limit
            )
            
            running_count = sum(1 for t in tasks if t['status'] in ['running', 'training'])
            pending_count = sum(1 for t in tasks if t['status'] == 'pending')
            paused_count = sum(1 for t in tasks if t['status'] == 'paused')
            
            return {
                'active_tasks': tasks,
                'total_count': len(tasks),
                'running_count': running_count,
                'pending_count': pending_count,
                'paused_count': paused_count
            }
            
        except Exception as e:
            logger.error(f"Failed to get active training tasks: {e}")
            return {
                'active_tasks': [],
                'total_count': 0,
                'running_count': 0,
                'pending_count': 0,
                'paused_count': 0
            }
    
    def get_training_duration_stats(
        self,
        user_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        days: int = 30
    ) -> Dict[str, Any]:
        """获取训练时长统计
        
        统计训练任务的平均、最小、最大时长及分布。
        
        Args:
            user_id: 用户ID
            tenant_id: 租户ID
            days: 统计天数
            
        Returns:
            Dict: 时长统计数据，包含：
                - avg_duration: 平均时长（小时）
                - min_duration: 最小时长（小时）
                - max_duration: 最大时长（小时）
                - total_count: 统计的任务数
                - duration_distribution: 时长分布
                
        Example:
            >>> stats = service.get_training_duration_stats(user_id="user123")
            >>> print(f"平均时长: {stats['avg_duration']:.2f}小时")
        """
        try:
            return self.repository.get_training_duration_stats(
                user_id=user_id,
                tenant_id=tenant_id,
                days=days
            )
        except Exception as e:
            logger.error(f"Failed to get training duration stats: {e}")
            return {
                'avg_duration': 0,
                'min_duration': 0,
                'max_duration': 0,
                'total_count': 0,
                'duration_distribution': []
            }
    
    def get_model_performance_distribution(
        self,
        user_id: Optional[str] = None,
        tenant_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """获取模型性能分布
        
        统计模型准确率分布和按类型的性能统计。
        
        Args:
            user_id: 用户ID
            tenant_id: 租户ID
            
        Returns:
            Dict: 性能分布数据，包含：
                - accuracy_distribution: 准确率分布
                - model_types: 按类型的统计
                - total_models: 模型总数
                
        Example:
            >>> perf = service.get_model_performance_distribution(user_id="user123")
            >>> for dist in perf['accuracy_distribution']:
            ...     print(f"{dist['range']}: {dist['count']}个模型")
        """
        try:
            return self.repository.get_model_performance_distribution(
                user_id=user_id,
                tenant_id=tenant_id
            )
        except Exception as e:
            logger.error(f"Failed to get model performance distribution: {e}")
            return {
                'accuracy_distribution': [],
                'model_types': [],
                'total_models': 0
            }
    
    def get_system_health_status(self) -> Dict[str, Any]:
        """获取系统健康状态
        
        检查各服务组件的运行状态。
        
        Returns:
            Dict: 系统健康状态，包含：
                - overall_status: 总体状态 (healthy/degraded/unhealthy)
                - services: 各服务状态
                - uptime: 运行时间
                - last_check: 最后检查时间
                
        Example:
            >>> health = service.get_system_health_status()
            >>> print(f"系统状态: {health['overall_status']}")
        """
        try:
            import time
            
            services = {}
            
            # 检查数据库
            try:
                from backend.modules.database.manager import get_database_manager
                db = get_database_manager()
                with db.get_db_session() as session:
                    from sqlalchemy import text
                    session.execute(text('SELECT 1'))
                services['database'] = 'running'
            except Exception as e:
                logger.warning(f"Database health check failed: {e}")
                services['database'] = 'error'
            
            # 检查 API 服务器（如果能执行到这里就是运行中）
            services['api_server'] = 'running'
            
            # 检查调度器（尝试导入）
            try:
                from backend.services.scheduler_service import get_scheduler_service
                services['scheduler'] = 'running'
            except Exception:
                services['scheduler'] = 'unknown'
            
            # 检查模型管理器
            try:
                from backend.services.model_service import ModelService
                services['model_manager'] = 'running'
            except Exception:
                services['model_manager'] = 'unknown'
            
            # 计算总体状态
            error_count = sum(1 for s in services.values() if s == 'error')
            unknown_count = sum(1 for s in services.values() if s == 'unknown')
            
            if error_count > 0:
                overall_status = 'unhealthy'
            elif unknown_count > 0:
                overall_status = 'degraded'
            else:
                overall_status = 'healthy'
            
            # 获取系统启动时间
            boot_time = psutil.boot_time()
            uptime_seconds = time.time() - boot_time
            days = int(uptime_seconds // 86400)
            hours = int((uptime_seconds % 86400) // 3600)
            minutes = int((uptime_seconds % 3600) // 60)
            uptime_str = f"{days}天 {hours}小时 {minutes}分钟"
            
            return {
                'overall_status': overall_status,
                'services': services,
                'uptime': uptime_str,
                'uptime_seconds': uptime_seconds,
                'last_check': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get system health status: {e}")
            return {
                'overall_status': 'unknown',
                'services': {},
                'uptime': 'unknown',
                'last_check': datetime.utcnow().isoformat()
            }
    
    def get_resource_usage_history(
        self,
        hours: int = 24,
        interval_minutes: int = 60
    ) -> List[Dict[str, Any]]:
        """获取资源使用历史
        
        获取系统资源的历史使用数据。
        
        Args:
            hours: 历史小时数
            interval_minutes: 数据点间隔（分钟）
            
        Returns:
            List[Dict]: 资源使用历史列表
            
        Note:
            由于没有持久化的监控数据，此方法返回基于当前值的模拟历史数据。
            实际生产环境应从时序数据库（如 Prometheus/InfluxDB）获取。
        """
        try:
            import random
            
            current = self._get_current_system_resources()
            history = []
            
            now = datetime.utcnow()
            points = hours * 60 // interval_minutes
            
            for i in range(points):
                timestamp = now - timedelta(minutes=i * interval_minutes)
                
                # 基于当前值生成历史数据（加入波动）
                history.append({
                    'timestamp': timestamp.isoformat(),
                    'cpu_percent': round(max(0, min(100, current.cpu_usage + random.uniform(-20, 20))), 2),
                    'memory_percent': round(max(0, min(100, current.memory_usage + random.uniform(-10, 10))), 2),
                    'gpu_percent': round(random.uniform(0, 100), 2),  # GPU 使用模拟
                    'disk_percent': round(max(0, min(100, current.disk_usage + random.uniform(-5, 5))), 2)
                })
            
            # 按时间升序排列
            history.reverse()
            
            return history
            
        except Exception as e:
            logger.error(f"Failed to get resource usage history: {e}")
            return []


# =============================================================================
# 全局服务实例
# =============================================================================

_dashboard_service: Optional[DashboardService] = None


def get_dashboard_service() -> DashboardService:
    """获取仪表盘服务单例
    
    Returns:
        DashboardService: 仪表盘服务实例
        
    Example:
        >>> service = get_dashboard_service()
        >>> overview = service.get_dashboard_overview()
    """
    global _dashboard_service
    if _dashboard_service is None:
        _dashboard_service = DashboardService()
    return _dashboard_service


def reset_dashboard_service():
    """重置仪表盘服务（用于测试）"""
    global _dashboard_service
    _dashboard_service = None

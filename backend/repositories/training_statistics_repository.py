# -*- coding: utf-8 -*-
"""训练统计仓库

提供训练统计数据访问接口，聚合查询训练任务、会话、进度等统计信息。

数据来源:
- TrainingJob: 训练任务
- TrainingSession: 训练会话
- TrainingProgress: 训练进度
- TrainingExecution: 训练执行
"""

import logging
import uuid
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
from sqlalchemy import func, and_, or_, desc, asc, extract, case

logger = logging.getLogger(__name__)


class TrainingStatisticsRepository:
    """
    训练统计仓库
    
    聚合查询多个训练相关表，提供统计数据访问。
    """
    
    def __init__(self, db_service=None, use_memory_storage: bool = False):
        """
        初始化仓库
        
        Args:
            db_service: 数据库服务实例
            use_memory_storage: 是否使用内存存储（用于测试）
        """
        self._use_memory_storage = use_memory_storage
        self._db_service = None
        
        # 内存存储 - 用于测试或无数据库环境
        self._memory_jobs: Dict[str, Dict] = {}
        self._memory_sessions: Dict[str, Dict] = {}
        self._memory_progress: Dict[str, List[Dict]] = {}
        
        if not use_memory_storage:
            self._init_database(db_service)
    
    def _init_database(self, db_service=None):
        """初始化数据库连接"""
        try:
            if db_service:
                self._db_service = db_service
            else:
                from backend.modules.database.manager import get_database_manager
                self._db_service = get_database_manager()
            logger.info("Database manager initialized for TrainingStatisticsRepository")
        except ImportError as e:
            logger.warning(f"Database manager not available: {e}, using memory storage")
            self._use_memory_storage = True
    
    # ==================== 基础统计 ====================
    
    def get_job_count_by_status(
        self,
        tenant_id: str,
        user_id: str = None,
        start_date: datetime = None,
        end_date: datetime = None
    ) -> Dict[str, int]:
        """
        按状态统计任务数量
        
        Args:
            tenant_id: 租户ID
            user_id: 用户ID（可选）
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            各状态的任务数量
        """
        if self._use_memory_storage:
            return self._get_job_count_by_status_memory(tenant_id, user_id, start_date, end_date)
        
        try:
            from backend.schemas.training_models import TrainingJob
            from backend.schemas.enums import TrainingStatus
            
            with self._db_service.get_session() as db:
                # 构建基础查询
                base_filters = [TrainingJob.tenant_id == tenant_id]
                
                if user_id:
                    base_filters.append(TrainingJob.user_id == user_id)
                
                if start_date:
                    base_filters.append(TrainingJob.created_at >= start_date)
                
                if end_date:
                    base_filters.append(TrainingJob.created_at <= end_date)
                
                # 聚合查询
                query = db.query(
                    TrainingJob.status,
                    func.count(TrainingJob.id).label('count')
                ).filter(
                    and_(*base_filters)
                ).group_by(TrainingJob.status)
                
                result = query.all()
                
                # 构建返回字典
                status_counts = {
                    'total': 0,
                    'pending': 0,
                    'running': 0,
                    'completed': 0,
                    'failed': 0,
                    'cancelled': 0,
                    'paused': 0
                }
                
                for status, count in result:
                    status_counts[status] = count
                    status_counts['total'] += count
                
                return status_counts
                
        except Exception as e:
            logger.error(f"Failed to get job count by status: {e}")
            return {'total': 0, 'pending': 0, 'running': 0, 'completed': 0, 
                    'failed': 0, 'cancelled': 0, 'paused': 0}
    
    def _get_job_count_by_status_memory(
        self,
        tenant_id: str,
        user_id: str = None,
        start_date: datetime = None,
        end_date: datetime = None
    ) -> Dict[str, int]:
        """内存模式下按状态统计任务数量"""
        status_counts = {
            'total': 0,
            'pending': 0,
            'running': 0,
            'completed': 0,
            'failed': 0,
            'cancelled': 0,
            'paused': 0
        }
        
        for job in self._memory_jobs.values():
            if job.get('tenant_id') != tenant_id:
                continue
            if user_id and job.get('user_id') != user_id:
                continue
            if start_date and job.get('created_at', datetime.min) < start_date:
                continue
            if end_date and job.get('created_at', datetime.max) > end_date:
                continue
            
            status = job.get('status', 'pending')
            if status in status_counts:
                status_counts[status] += 1
            status_counts['total'] += 1
        
        return status_counts
    
    def get_training_time_statistics(
        self,
        tenant_id: str,
        user_id: str = None,
        start_date: datetime = None,
        end_date: datetime = None
    ) -> Dict[str, Any]:
        """
        获取训练时间统计
        
        Args:
            tenant_id: 租户ID
            user_id: 用户ID
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            训练时间统计（平均时长、总时长、最长、最短）
        """
        if self._use_memory_storage:
            return self._get_training_time_statistics_memory(tenant_id, user_id)
        
        try:
            from backend.schemas.training_models import TrainingJob
            
            with self._db_service.get_session() as db:
                base_filters = [
                    TrainingJob.tenant_id == tenant_id,
                    TrainingJob.completed_at.isnot(None),
                    TrainingJob.started_at.isnot(None)
                ]
                
                if user_id:
                    base_filters.append(TrainingJob.user_id == user_id)
                
                if start_date:
                    base_filters.append(TrainingJob.created_at >= start_date)
                
                if end_date:
                    base_filters.append(TrainingJob.created_at <= end_date)
                
                # 获取已完成任务的时间信息
                jobs = db.query(
                    TrainingJob.started_at,
                    TrainingJob.completed_at
                ).filter(
                    and_(*base_filters)
                ).all()
                
                if not jobs:
                    return {
                        'average_duration_seconds': 0,
                        'total_duration_seconds': 0,
                        'total_duration_hours': 0.0,
                        'max_duration_seconds': 0,
                        'min_duration_seconds': 0,
                        'completed_jobs_count': 0
                    }
                
                durations = []
                for started_at, completed_at in jobs:
                    if started_at and completed_at:
                        duration = (completed_at - started_at).total_seconds()
                        durations.append(duration)
                
                if not durations:
                    return {
                        'average_duration_seconds': 0,
                        'total_duration_seconds': 0,
                        'total_duration_hours': 0.0,
                        'max_duration_seconds': 0,
                        'min_duration_seconds': 0,
                        'completed_jobs_count': 0
                    }
                
                total_duration = sum(durations)
                
                return {
                    'average_duration_seconds': total_duration / len(durations),
                    'total_duration_seconds': total_duration,
                    'total_duration_hours': total_duration / 3600,
                    'max_duration_seconds': max(durations),
                    'min_duration_seconds': min(durations),
                    'completed_jobs_count': len(durations)
                }
                
        except Exception as e:
            logger.error(f"Failed to get training time statistics: {e}")
            return {
                'average_duration_seconds': 0,
                'total_duration_seconds': 0,
                'total_duration_hours': 0.0,
                'max_duration_seconds': 0,
                'min_duration_seconds': 0,
                'completed_jobs_count': 0
            }
    
    def _get_training_time_statistics_memory(
        self,
        tenant_id: str,
        user_id: str = None
    ) -> Dict[str, Any]:
        """内存模式下获取训练时间统计"""
        durations = []
        
        for job in self._memory_jobs.values():
            if job.get('tenant_id') != tenant_id:
                continue
            if user_id and job.get('user_id') != user_id:
                continue
            
            started_at = job.get('started_at')
            completed_at = job.get('completed_at')
            
            if started_at and completed_at:
                duration = (completed_at - started_at).total_seconds()
                durations.append(duration)
        
        if not durations:
            return {
                'average_duration_seconds': 0,
                'total_duration_seconds': 0,
                'total_duration_hours': 0.0,
                'max_duration_seconds': 0,
                'min_duration_seconds': 0,
                'completed_jobs_count': 0
            }
        
        total_duration = sum(durations)
        
        return {
            'average_duration_seconds': total_duration / len(durations),
            'total_duration_seconds': total_duration,
            'total_duration_hours': total_duration / 3600,
            'max_duration_seconds': max(durations),
            'min_duration_seconds': min(durations),
            'completed_jobs_count': len(durations)
        }
    
    # ==================== 资源使用统计 ====================
    
    def get_resource_usage_statistics(
        self,
        tenant_id: str,
        user_id: str = None,
        start_date: datetime = None,
        end_date: datetime = None
    ) -> Dict[str, Any]:
        """
        获取资源使用统计
        
        Args:
            tenant_id: 租户ID
            user_id: 用户ID
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            资源使用统计
        """
        if self._use_memory_storage:
            return self._get_resource_usage_statistics_memory(tenant_id, user_id)
        
        try:
            from backend.schemas.training_models import TrainingProgress
            
            with self._db_service.get_session() as db:
                # 获取进度记录中的资源使用数据
                query = db.query(
                    func.avg(TrainingProgress.gpu_utilization).label('avg_gpu'),
                    func.avg(TrainingProgress.gpu_memory_used).label('avg_gpu_memory'),
                    func.avg(TrainingProgress.cpu_utilization).label('avg_cpu'),
                    func.avg(TrainingProgress.cpu_memory_used).label('avg_cpu_memory'),
                    func.max(TrainingProgress.gpu_utilization).label('max_gpu'),
                    func.max(TrainingProgress.cpu_utilization).label('max_cpu')
                )
                
                # 添加时间过滤
                filters = []
                if start_date:
                    filters.append(TrainingProgress.created_at >= start_date)
                if end_date:
                    filters.append(TrainingProgress.created_at <= end_date)
                
                if filters:
                    query = query.filter(and_(*filters))
                
                result = query.first()
                
                return {
                    'gpu': {
                        'average_utilization': float(result.avg_gpu or 0),
                        'max_utilization': float(result.max_gpu or 0),
                        'average_memory_used_gb': float(result.avg_gpu_memory or 0)
                    },
                    'cpu': {
                        'average_utilization': float(result.avg_cpu or 0),
                        'max_utilization': float(result.max_cpu or 0),
                        'average_memory_used_gb': float(result.avg_cpu_memory or 0)
                    }
                }
                
        except Exception as e:
            logger.error(f"Failed to get resource usage statistics: {e}")
            return {
                'gpu': {'average_utilization': 0, 'max_utilization': 0, 'average_memory_used_gb': 0},
                'cpu': {'average_utilization': 0, 'max_utilization': 0, 'average_memory_used_gb': 0}
            }
    
    def _get_resource_usage_statistics_memory(
        self,
        tenant_id: str,
        user_id: str = None
    ) -> Dict[str, Any]:
        """内存模式下获取资源使用统计"""
        gpu_utils = []
        cpu_utils = []
        gpu_memory = []
        cpu_memory = []
        
        for session_id, progress_list in self._memory_progress.items():
            for progress in progress_list:
                if progress.get('gpu_utilization'):
                    gpu_utils.append(progress['gpu_utilization'])
                if progress.get('cpu_utilization'):
                    cpu_utils.append(progress['cpu_utilization'])
                if progress.get('gpu_memory_used'):
                    gpu_memory.append(progress['gpu_memory_used'])
                if progress.get('cpu_memory_used'):
                    cpu_memory.append(progress['cpu_memory_used'])
        
        return {
            'gpu': {
                'average_utilization': sum(gpu_utils) / len(gpu_utils) if gpu_utils else 0,
                'max_utilization': max(gpu_utils) if gpu_utils else 0,
                'average_memory_used_gb': sum(gpu_memory) / len(gpu_memory) if gpu_memory else 0
            },
            'cpu': {
                'average_utilization': sum(cpu_utils) / len(cpu_utils) if cpu_utils else 0,
                'max_utilization': max(cpu_utils) if cpu_utils else 0,
                'average_memory_used_gb': sum(cpu_memory) / len(cpu_memory) if cpu_memory else 0
            }
        }
    
    # ==================== 趋势统计 ====================
    
    def get_daily_statistics(
        self,
        tenant_id: str,
        days: int = 30,
        user_id: str = None
    ) -> List[Dict[str, Any]]:
        """
        获取每日统计数据
        
        Args:
            tenant_id: 租户ID
            days: 统计天数
            user_id: 用户ID
            
        Returns:
            每日统计列表
        """
        if self._use_memory_storage:
            return self._get_daily_statistics_memory(tenant_id, days, user_id)
        
        try:
            from backend.schemas.training_models import TrainingJob
            
            with self._db_service.get_session() as db:
                end_date = datetime.utcnow()
                start_date = end_date - timedelta(days=days)
                
                base_filters = [
                    TrainingJob.tenant_id == tenant_id,
                    TrainingJob.created_at >= start_date,
                    TrainingJob.created_at <= end_date
                ]
                
                if user_id:
                    base_filters.append(TrainingJob.user_id == user_id)
                
                # 按日期分组统计
                query = db.query(
                    func.date(TrainingJob.created_at).label('date'),
                    func.count(TrainingJob.id).label('total'),
                    func.count(case(
                        (TrainingJob.status == 'completed', 1)
                    )).label('completed'),
                    func.count(case(
                        (TrainingJob.status == 'failed', 1)
                    )).label('failed'),
                    func.count(case(
                        (TrainingJob.status == 'running', 1)
                    )).label('running')
                ).filter(
                    and_(*base_filters)
                ).group_by(
                    func.date(TrainingJob.created_at)
                ).order_by(
                    func.date(TrainingJob.created_at)
                )
                
                results = query.all()
                
                # 构建完整的日期范围
                daily_stats = []
                current_date = start_date.date()
                
                # 转换查询结果为字典
                result_dict = {}
                for row in results:
                    date_str = str(row.date)
                    result_dict[date_str] = {
                        'total': row.total,
                        'completed': row.completed,
                        'failed': row.failed,
                        'running': row.running
                    }
                
                # 填充所有日期
                while current_date <= end_date.date():
                    date_str = str(current_date)
                    if date_str in result_dict:
                        daily_stats.append({
                            'date': date_str,
                            **result_dict[date_str],
                            'success_rate': (result_dict[date_str]['completed'] / 
                                           result_dict[date_str]['total'] * 100 
                                           if result_dict[date_str]['total'] > 0 else 0)
                        })
                    else:
                        daily_stats.append({
                            'date': date_str,
                            'total': 0,
                            'completed': 0,
                            'failed': 0,
                            'running': 0,
                            'success_rate': 0
                        })
                    current_date += timedelta(days=1)
                
                return daily_stats
                
        except Exception as e:
            logger.error(f"Failed to get daily statistics: {e}")
            return []
    
    def _get_daily_statistics_memory(
        self,
        tenant_id: str,
        days: int = 30,
        user_id: str = None
    ) -> List[Dict[str, Any]]:
        """内存模式下获取每日统计"""
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)
        
        # 按日期分组
        daily_data = defaultdict(lambda: {'total': 0, 'completed': 0, 'failed': 0, 'running': 0})
        
        for job in self._memory_jobs.values():
            if job.get('tenant_id') != tenant_id:
                continue
            if user_id and job.get('user_id') != user_id:
                continue
            
            created_at = job.get('created_at')
            if not created_at or created_at < start_date or created_at > end_date:
                continue
            
            date_str = created_at.strftime('%Y-%m-%d')
            daily_data[date_str]['total'] += 1
            
            status = job.get('status', 'pending')
            if status in daily_data[date_str]:
                daily_data[date_str][status] += 1
        
        # 构建完整的日期范围
        daily_stats = []
        current_date = start_date.date()
        
        while current_date <= end_date.date():
            date_str = str(current_date)
            data = daily_data.get(date_str, {'total': 0, 'completed': 0, 'failed': 0, 'running': 0})
            daily_stats.append({
                'date': date_str,
                **data,
                'success_rate': (data['completed'] / data['total'] * 100 if data['total'] > 0 else 0)
            })
            current_date += timedelta(days=1)
        
        return daily_stats
    
    # ==================== 模型/场景统计 ====================
    
    def get_model_usage_statistics(
        self,
        tenant_id: str,
        user_id: str = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        获取模型使用统计
        
        Args:
            tenant_id: 租户ID
            user_id: 用户ID
            limit: 返回数量限制
            
        Returns:
            模型使用统计列表
        """
        if self._use_memory_storage:
            return self._get_model_usage_statistics_memory(tenant_id, user_id, limit)
        
        try:
            from backend.schemas.training_models import TrainingJob
            
            with self._db_service.get_session() as db:
                base_filters = [
                    TrainingJob.tenant_id == tenant_id,
                    TrainingJob.model_name.isnot(None)
                ]
                
                if user_id:
                    base_filters.append(TrainingJob.user_id == user_id)
                
                query = db.query(
                    TrainingJob.model_name,
                    func.count(TrainingJob.id).label('usage_count'),
                    func.count(case(
                        (TrainingJob.status == 'completed', 1)
                    )).label('success_count')
                ).filter(
                    and_(*base_filters)
                ).group_by(
                    TrainingJob.model_name
                ).order_by(
                    desc(func.count(TrainingJob.id))
                ).limit(limit)
                
                results = query.all()
                
                return [
                    {
                        'model_name': row.model_name,
                        'usage_count': row.usage_count,
                        'success_count': row.success_count,
                        'success_rate': (row.success_count / row.usage_count * 100 
                                        if row.usage_count > 0 else 0)
                    }
                    for row in results
                ]
                
        except Exception as e:
            logger.error(f"Failed to get model usage statistics: {e}")
            return []
    
    def _get_model_usage_statistics_memory(
        self,
        tenant_id: str,
        user_id: str = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """内存模式下获取模型使用统计"""
        model_stats = defaultdict(lambda: {'usage_count': 0, 'success_count': 0})
        
        for job in self._memory_jobs.values():
            if job.get('tenant_id') != tenant_id:
                continue
            if user_id and job.get('user_id') != user_id:
                continue
            
            model_name = job.get('model_name')
            if not model_name:
                continue
            
            model_stats[model_name]['usage_count'] += 1
            if job.get('status') == 'completed':
                model_stats[model_name]['success_count'] += 1
        
        # 排序并限制数量
        sorted_stats = sorted(
            model_stats.items(),
            key=lambda x: x[1]['usage_count'],
            reverse=True
        )[:limit]
        
        return [
            {
                'model_name': model_name,
                'usage_count': stats['usage_count'],
                'success_count': stats['success_count'],
                'success_rate': (stats['success_count'] / stats['usage_count'] * 100 
                                if stats['usage_count'] > 0 else 0)
            }
            for model_name, stats in sorted_stats
        ]
    
    def get_scenario_statistics(
        self,
        tenant_id: str,
        user_id: str = None
    ) -> List[Dict[str, Any]]:
        """
        获取训练场景统计
        
        Args:
            tenant_id: 租户ID
            user_id: 用户ID
            
        Returns:
            场景统计列表
        """
        if self._use_memory_storage:
            return self._get_scenario_statistics_memory(tenant_id, user_id)
        
        try:
            from backend.schemas.training_models import TrainingJob
            
            with self._db_service.get_session() as db:
                base_filters = [
                    TrainingJob.tenant_id == tenant_id,
                    TrainingJob.scenario_type.isnot(None)
                ]
                
                if user_id:
                    base_filters.append(TrainingJob.user_id == user_id)
                
                query = db.query(
                    TrainingJob.scenario_type,
                    func.count(TrainingJob.id).label('count'),
                    func.count(case(
                        (TrainingJob.status == 'completed', 1)
                    )).label('completed'),
                    func.count(case(
                        (TrainingJob.status == 'failed', 1)
                    )).label('failed')
                ).filter(
                    and_(*base_filters)
                ).group_by(
                    TrainingJob.scenario_type
                ).order_by(
                    desc(func.count(TrainingJob.id))
                )
                
                results = query.all()
                
                return [
                    {
                        'scenario_type': row.scenario_type,
                        'total_count': row.count,
                        'completed_count': row.completed,
                        'failed_count': row.failed,
                        'success_rate': (row.completed / row.count * 100 if row.count > 0 else 0)
                    }
                    for row in results
                ]
                
        except Exception as e:
            logger.error(f"Failed to get scenario statistics: {e}")
            return []
    
    def _get_scenario_statistics_memory(
        self,
        tenant_id: str,
        user_id: str = None
    ) -> List[Dict[str, Any]]:
        """内存模式下获取场景统计"""
        scenario_stats = defaultdict(lambda: {'count': 0, 'completed': 0, 'failed': 0})
        
        for job in self._memory_jobs.values():
            if job.get('tenant_id') != tenant_id:
                continue
            if user_id and job.get('user_id') != user_id:
                continue
            
            scenario_type = job.get('scenario_type')
            if not scenario_type:
                continue
            
            scenario_stats[scenario_type]['count'] += 1
            
            status = job.get('status')
            if status == 'completed':
                scenario_stats[scenario_type]['completed'] += 1
            elif status == 'failed':
                scenario_stats[scenario_type]['failed'] += 1
        
        return [
            {
                'scenario_type': scenario_type,
                'total_count': stats['count'],
                'completed_count': stats['completed'],
                'failed_count': stats['failed'],
                'success_rate': (stats['completed'] / stats['count'] * 100 if stats['count'] > 0 else 0)
            }
            for scenario_type, stats in sorted(
                scenario_stats.items(),
                key=lambda x: x[1]['count'],
                reverse=True
            )
        ]
    
    # ==================== 性能指标统计 ====================
    
    def get_performance_metrics_statistics(
        self,
        tenant_id: str,
        user_id: str = None,
        metric_name: str = None
    ) -> Dict[str, Any]:
        """
        获取性能指标统计
        
        Args:
            tenant_id: 租户ID
            user_id: 用户ID
            metric_name: 指标名称过滤
            
        Returns:
            性能指标统计
        """
        if self._use_memory_storage:
            return self._get_performance_metrics_statistics_memory(tenant_id, user_id, metric_name)
        
        try:
            from backend.schemas.training_models import TrainingProgress
            
            with self._db_service.get_session() as db:
                query = db.query(
                    func.avg(TrainingProgress.loss).label('avg_loss'),
                    func.min(TrainingProgress.loss).label('min_loss'),
                    func.avg(TrainingProgress.accuracy).label('avg_accuracy'),
                    func.max(TrainingProgress.accuracy).label('max_accuracy'),
                    func.avg(TrainingProgress.samples_per_second).label('avg_throughput'),
                    func.max(TrainingProgress.samples_per_second).label('max_throughput')
                )
                
                result = query.first()
                
                return {
                    'loss': {
                        'average': float(result.avg_loss or 0),
                        'minimum': float(result.min_loss or 0)
                    },
                    'accuracy': {
                        'average': float(result.avg_accuracy or 0),
                        'maximum': float(result.max_accuracy or 0)
                    },
                    'throughput': {
                        'average_samples_per_second': float(result.avg_throughput or 0),
                        'max_samples_per_second': float(result.max_throughput or 0)
                    }
                }
                
        except Exception as e:
            logger.error(f"Failed to get performance metrics statistics: {e}")
            return {
                'loss': {'average': 0, 'minimum': 0},
                'accuracy': {'average': 0, 'maximum': 0},
                'throughput': {'average_samples_per_second': 0, 'max_samples_per_second': 0}
            }
    
    def _get_performance_metrics_statistics_memory(
        self,
        tenant_id: str,
        user_id: str = None,
        metric_name: str = None
    ) -> Dict[str, Any]:
        """内存模式下获取性能指标统计"""
        losses = []
        accuracies = []
        throughputs = []
        
        for session_id, progress_list in self._memory_progress.items():
            for progress in progress_list:
                if progress.get('loss'):
                    losses.append(progress['loss'])
                if progress.get('accuracy'):
                    accuracies.append(progress['accuracy'])
                if progress.get('samples_per_second'):
                    throughputs.append(progress['samples_per_second'])
        
        return {
            'loss': {
                'average': sum(losses) / len(losses) if losses else 0,
                'minimum': min(losses) if losses else 0
            },
            'accuracy': {
                'average': sum(accuracies) / len(accuracies) if accuracies else 0,
                'maximum': max(accuracies) if accuracies else 0
            },
            'throughput': {
                'average_samples_per_second': sum(throughputs) / len(throughputs) if throughputs else 0,
                'max_samples_per_second': max(throughputs) if throughputs else 0
            }
        }
    
    # ==================== 汇总统计 ====================
    
    def get_comprehensive_statistics(
        self,
        tenant_id: str,
        user_id: str = None,
        days: int = 30
    ) -> Dict[str, Any]:
        """
        获取综合统计信息
        
        Args:
            tenant_id: 租户ID
            user_id: 用户ID
            days: 统计天数
            
        Returns:
            综合统计信息
        """
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)
        
        # 聚合所有统计
        job_counts = self.get_job_count_by_status(tenant_id, user_id, start_date, end_date)
        time_stats = self.get_training_time_statistics(tenant_id, user_id, start_date, end_date)
        resource_stats = self.get_resource_usage_statistics(tenant_id, user_id, start_date, end_date)
        model_stats = self.get_model_usage_statistics(tenant_id, user_id, limit=5)
        scenario_stats = self.get_scenario_statistics(tenant_id, user_id)
        performance_stats = self.get_performance_metrics_statistics(tenant_id, user_id)
        
        # 计算成功率
        total = job_counts.get('total', 0)
        completed = job_counts.get('completed', 0)
        success_rate = (completed / total * 100) if total > 0 else 0
        
        return {
            'period': {
                'start_date': start_date.isoformat(),
                'end_date': end_date.isoformat(),
                'days': days
            },
            'summary': {
                'total_jobs': total,
                'completed_jobs': completed,
                'failed_jobs': job_counts.get('failed', 0),
                'running_jobs': job_counts.get('running', 0),
                'pending_jobs': job_counts.get('pending', 0),
                'cancelled_jobs': job_counts.get('cancelled', 0),
                'paused_jobs': job_counts.get('paused', 0),
                'success_rate': round(success_rate, 2)
            },
            'time_statistics': time_stats,
            'resource_usage': resource_stats,
            'top_models': model_stats,
            'scenario_breakdown': scenario_stats,
            'performance_metrics': performance_stats,
            'timestamp': datetime.utcnow().isoformat()
        }
    
    # ==================== 内存数据管理（用于测试） ====================
    
    def add_memory_job(self, job_data: Dict[str, Any]) -> str:
        """添加内存中的任务数据（用于测试）"""
        job_id = job_data.get('job_id') or str(uuid.uuid4())
        job_data['job_id'] = job_id
        if 'created_at' not in job_data:
            job_data['created_at'] = datetime.utcnow()
        self._memory_jobs[job_id] = job_data
        return job_id
    
    def add_memory_session(self, session_data: Dict[str, Any]) -> str:
        """添加内存中的会话数据（用于测试）"""
        session_id = session_data.get('session_id') or str(uuid.uuid4())
        session_data['session_id'] = session_id
        self._memory_sessions[session_id] = session_data
        return session_id
    
    def add_memory_progress(self, session_id: str, progress_data: Dict[str, Any]) -> None:
        """添加内存中的进度数据（用于测试）"""
        if session_id not in self._memory_progress:
            self._memory_progress[session_id] = []
        self._memory_progress[session_id].append(progress_data)
    
    def clear_memory_data(self) -> None:
        """清空内存数据（用于测试）"""
        self._memory_jobs.clear()
        self._memory_sessions.clear()
        self._memory_progress.clear()


# ==================== 全局实例管理 ====================

_training_statistics_repository: Optional[TrainingStatisticsRepository] = None


def get_training_statistics_repository(
    use_memory_storage: bool = False
) -> TrainingStatisticsRepository:
    """
    获取训练统计仓库实例
    
    Args:
        use_memory_storage: 是否使用内存存储
        
    Returns:
        仓库实例
    """
    global _training_statistics_repository
    
    if _training_statistics_repository is None:
        _training_statistics_repository = TrainingStatisticsRepository(
            use_memory_storage=use_memory_storage
        )
    
    return _training_statistics_repository


def reset_training_statistics_repository() -> None:
    """重置全局仓库实例（用于测试）"""
    global _training_statistics_repository
    _training_statistics_repository = None


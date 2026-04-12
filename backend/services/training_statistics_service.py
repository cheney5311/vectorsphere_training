# -*- coding: utf-8 -*-
"""训练统计服务

提供训练任务的统计信息，支持基础统计、详细统计、趋势分析、资源监控等功能。

功能特性:
- 基础统计：任务数量、成功率、运行状态
- 详细统计：时间分析、资源使用、模型分布
- 趋势分析：每日/每周/每月趋势
- 实时监控：运行中任务状态
- 资源统计：GPU/CPU使用情况

架构调用关系：
API层 -> Service层 (本模块) -> Launcher层 (training_launcher.py) / Repository层
"""

import sys
import os
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from backend.core.exceptions import BusinessLogicError, ValidationError

logger = logging.getLogger(__name__)

# =============================================================================
# 训练启动器集成 (用于获取状态和诊断信息)
# =============================================================================

from backend.modules.training.launcher import (
    get_module_availability,
    diagnose_launcher_module,
    get_all_training_modes,
)

from backend.modules.training.progress import (
    get_progress_manager,
    TrainingProgressManager,
)


class StatisticsTimeRange(Enum):
    """统计时间范围"""
    LAST_24_HOURS = "last_24_hours"
    LAST_7_DAYS = "last_7_days"
    LAST_30_DAYS = "last_30_days"
    LAST_90_DAYS = "last_90_days"
    THIS_MONTH = "this_month"
    THIS_YEAR = "this_year"
    CUSTOM = "custom"


class StatisticsGroupBy(Enum):
    """统计分组方式"""
    HOUR = "hour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"


@dataclass
class StatisticsFilter:
    """统计过滤条件"""
    tenant_id: str
    user_id: Optional[str] = None
    time_range: StatisticsTimeRange = StatisticsTimeRange.LAST_30_DAYS
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    scenario_types: List[str] = field(default_factory=list)
    model_names: List[str] = field(default_factory=list)
    statuses: List[str] = field(default_factory=list)
    
    def get_date_range(self) -> tuple:
        """获取日期范围"""
        end_date = datetime.utcnow()
        
        if self.time_range == StatisticsTimeRange.CUSTOM:
            return self.start_date, self.end_date or end_date
        
        range_mapping = {
            StatisticsTimeRange.LAST_24_HOURS: timedelta(hours=24),
            StatisticsTimeRange.LAST_7_DAYS: timedelta(days=7),
            StatisticsTimeRange.LAST_30_DAYS: timedelta(days=30),
            StatisticsTimeRange.LAST_90_DAYS: timedelta(days=90),
        }
        
        if self.time_range == StatisticsTimeRange.THIS_MONTH:
            start_date = end_date.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        elif self.time_range == StatisticsTimeRange.THIS_YEAR:
            start_date = end_date.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
        else:
            delta = range_mapping.get(self.time_range, timedelta(days=30))
            start_date = end_date - delta
        
        return start_date, end_date


class TrainingStatisticsService:
    """训练统计服务
    
    提供全面的训练统计功能，包括基础统计、详细统计、趋势分析等。
    """
    
    def __init__(self, use_memory_storage: bool = False):
        """初始化服务
        
        Args:
            use_memory_storage: 是否使用内存存储模式
        """
        self._use_memory_storage = use_memory_storage
        self._repository = None
        self._scenario_manager = None
        self._service_start_time = datetime.utcnow()
        
        self._init_dependencies()
    
    def _init_dependencies(self):
        """初始化依赖"""
        try:
            from backend.repositories.training_statistics_repository import (
                get_training_statistics_repository
            )
            self._repository = get_training_statistics_repository(
                use_memory_storage=self._use_memory_storage
            )
            logger.info("Training statistics repository initialized")
        except ImportError as e:
            logger.warning(f"Failed to import repository: {e}")
            self._repository = None
        
        try:
            from backend.modules.training.scenarios.scenario_manager import get_scenario_manager
            self._scenario_manager = get_scenario_manager()
            logger.info("Scenario manager initialized")
        except ImportError as e:
            logger.warning(f"Failed to import scenario manager: {e}")
            self._scenario_manager = None
        
        # 初始化启动器相关
        try:
            availability = get_module_availability()
            logger.info(f"Launcher module availability: {availability}")
        except Exception as e:
            logger.warning(f"Failed to check launcher availability: {e}")
        
        # 初始化进度管理器
        try:
            self._progress_manager = get_progress_manager()
            logger.info("Progress manager initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize progress manager: {e}")
            self._progress_manager = None
    
    # ==================== 基础统计 ====================
    
    def get_basic_statistics(
        self,
        tenant_id: str = None,
        user_id: str = None
    ) -> Dict[str, Any]:
        """获取基础统计信息
        
        Args:
            tenant_id: 租户ID
            user_id: 用户ID
            
        Returns:
            基础统计信息
        """
        try:
            # 使用仓库层获取数据
            if self._repository and tenant_id:
                job_counts = self._repository.get_job_count_by_status(
                    tenant_id=tenant_id,
                    user_id=user_id
                )
                time_stats = self._repository.get_training_time_statistics(
                    tenant_id=tenant_id,
                    user_id=user_id
                )
                
                total = job_counts.get('total', 0)
                completed = job_counts.get('completed', 0)
                failed = job_counts.get('failed', 0)
                
                # 计算成功率
                success_rate = 0.0
                if total > 0:
                    success_rate = round(completed / total, 4)
                
                return {
                    'total_jobs': total,
                    'completed_jobs': completed,
                    'failed_jobs': failed,
                    'running_jobs': job_counts.get('running', 0),
                    'pending_jobs': job_counts.get('pending', 0),
                    'paused_jobs': job_counts.get('paused', 0),
                    'cancelled_jobs': job_counts.get('cancelled', 0),
                    'success_rate': success_rate,
                    'average_duration': time_stats.get('average_duration_seconds', 0),
                    'total_training_hours': time_stats.get('total_duration_hours', 0),
                    'timestamp': datetime.utcnow().isoformat()
                }
            
            # 尝试从场景管理器获取运行时统计
            if self._scenario_manager:
                try:
                    manager_stats = self._scenario_manager.get_statistics()
                    return {
                        'total_jobs': manager_stats.get('total_jobs', 0),
                        'completed_jobs': manager_stats.get('completed_jobs', 0),
                        'failed_jobs': manager_stats.get('failed_jobs', 0),
                        'running_jobs': manager_stats.get('running_jobs', 0),
                        'pending_jobs': manager_stats.get('pending_jobs', 0),
                        'paused_jobs': 0,
                        'cancelled_jobs': 0,
                        'success_rate': self._calculate_success_rate(manager_stats),
                        'average_duration': 0,
                        'total_training_hours': 0,
                        'timestamp': datetime.utcnow().isoformat()
                    }
                except Exception as e:
                    logger.warning(f"Failed to get stats from scenario manager: {e}")
            
            # 返回默认统计
            return self._get_default_basic_statistics()
            
        except Exception as e:
            logger.error(f"Failed to get basic statistics: {e}")
            raise BusinessLogicError(
                f"获取基础统计信息失败: {e}",
                operation="get_basic_statistics"
            )
    
    def _get_default_basic_statistics(self) -> Dict[str, Any]:
        """返回默认的基础统计"""
        return {
            'total_jobs': 0,
            'completed_jobs': 0,
            'failed_jobs': 0,
            'running_jobs': 0,
            'pending_jobs': 0,
            'paused_jobs': 0,
            'cancelled_jobs': 0,
            'success_rate': 0.0,
            'average_duration': 0,
            'total_training_hours': 0.0,
            'timestamp': datetime.utcnow().isoformat()
        }
    
    # ==================== 详细统计 ====================
    
    def get_detailed_statistics(
        self,
        tenant_id: str = None,
        user_id: str = None,
        days: int = 30
    ) -> Dict[str, Any]:
        """获取详细统计信息
        
        Args:
            tenant_id: 租户ID
            user_id: 用户ID
            days: 统计天数
            
        Returns:
            详细统计信息
        """
        try:
            # 使用仓库层获取数据
            if self._repository and tenant_id:
                comprehensive = self._repository.get_comprehensive_statistics(
                    tenant_id=tenant_id,
                    user_id=user_id,
                    days=days
                )
                
                daily_stats = self._repository.get_daily_statistics(
                    tenant_id=tenant_id,
                    days=days,
                    user_id=user_id
                )
                
                summary = comprehensive.get('summary', {})
                time_stats = comprehensive.get('time_statistics', {})
                resource_stats = comprehensive.get('resource_usage', {})
                top_models = comprehensive.get('top_models', [])
                
                # 获取最常用的模型
                most_used_model = top_models[0]['model_name'] if top_models else 'N/A'
                
                return {
                    'total_jobs': summary.get('total_jobs', 0),
                    'running_jobs': summary.get('running_jobs', 0),
                    'completed_jobs': summary.get('completed_jobs', 0),
                    'failed_jobs': summary.get('failed_jobs', 0),
                    'cancelled_jobs': summary.get('cancelled_jobs', 0),
                    'paused_jobs': summary.get('paused_jobs', 0),
                    'pending_jobs': summary.get('pending_jobs', 0),
                    'success_rate': summary.get('success_rate', 0) / 100,
                    'average_training_time': time_stats.get('average_duration_seconds', 0),
                    'total_training_hours': time_stats.get('total_duration_hours', 0),
                    'max_training_time': time_stats.get('max_duration_seconds', 0),
                    'min_training_time': time_stats.get('min_duration_seconds', 0),
                    'most_used_model': most_used_model,
                    'resource_usage': {
                        'cpu_avg': resource_stats.get('cpu', {}).get('average_utilization', 0),
                        'memory_avg': resource_stats.get('cpu', {}).get('average_memory_used_gb', 0),
                        'gpu_avg': resource_stats.get('gpu', {}).get('average_utilization', 0),
                        'gpu_memory_avg': resource_stats.get('gpu', {}).get('average_memory_used_gb', 0)
                    },
                    'daily_stats': [
                        {
                            'date': stat['date'],
                            'jobs_count': stat['total'],
                            'success_count': stat['completed'],
                            'failed_count': stat['failed'],
                            'success_rate': stat['success_rate']
                        }
                        for stat in daily_stats[-7:]  # 最近7天
                    ],
                    'top_models': top_models,
                    'scenario_breakdown': comprehensive.get('scenario_breakdown', []),
                    'performance_metrics': comprehensive.get('performance_metrics', {}),
                    'timestamp': datetime.utcnow().isoformat()
                }
            
            # 返回默认详细统计
            return self._get_default_detailed_statistics()
            
        except Exception as e:
            logger.error(f"Failed to get detailed statistics: {e}")
            raise BusinessLogicError(
                f"获取详细统计信息失败: {e}",
                operation="get_detailed_statistics"
            )
    
    def _get_default_detailed_statistics(self) -> Dict[str, Any]:
        """返回默认的详细统计"""
        return {
            'total_jobs': 0,
            'running_jobs': 0,
            'completed_jobs': 0,
            'failed_jobs': 0,
            'cancelled_jobs': 0,
            'paused_jobs': 0,
            'pending_jobs': 0,
            'success_rate': 0.0,
            'average_training_time': 0,
            'total_training_hours': 0.0,
            'most_used_model': 'N/A',
            'resource_usage': {
                'cpu_avg': 0,
                'memory_avg': 0,
                'gpu_avg': 0
            },
            'daily_stats': [],
            'top_models': [],
            'scenario_breakdown': [],
            'performance_metrics': {},
            'timestamp': datetime.utcnow().isoformat()
        }
    
    # ==================== 概览统计 ====================
    
    def get_statistics_overview(
        self,
        tenant_id: str = None,
        user_id: str = None
    ) -> Dict[str, Any]:
        """获取统计概览信息
        
        Args:
            tenant_id: 租户ID
            user_id: 用户ID
            
        Returns:
            统计概览信息
        """
        try:
            basic_stats = self.get_basic_statistics(tenant_id, user_id)
            detailed_stats = self.get_detailed_statistics(tenant_id, user_id, days=7)
            
            return {
                'overall_stats': basic_stats,
                'recent_trends': detailed_stats.get('daily_stats', []),
                'performance_metrics': {
                    'success_rate': basic_stats.get('success_rate', 0),
                    'average_duration': basic_stats.get('average_duration', 0),
                    'resource_usage': detailed_stats.get('resource_usage', {})
                },
                'top_models': detailed_stats.get('top_models', []),
                'scenario_breakdown': detailed_stats.get('scenario_breakdown', []),
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get statistics overview: {e}")
            raise BusinessLogicError(
                f"获取统计概览失败: {e}",
                operation="get_statistics_overview"
            )
    
    # ==================== 趋势统计 ====================
    
    def get_trend_statistics(
        self,
        tenant_id: str = None,
        user_id: str = None,
        days: int = 7,
        group_by: StatisticsGroupBy = StatisticsGroupBy.DAY
    ) -> Dict[str, Any]:
        """获取趋势统计信息
        
        Args:
            tenant_id: 租户ID
            user_id: 用户ID
            days: 统计天数
            group_by: 分组方式
            
        Returns:
            趋势统计信息
        """
        try:
            if self._repository and tenant_id:
                daily_stats = self._repository.get_daily_statistics(
                    tenant_id=tenant_id,
                    days=days,
                    user_id=user_id
                )
                
                job_counts = self._repository.get_job_count_by_status(
                    tenant_id=tenant_id,
                    user_id=user_id
                )
                
                # 计算趋势
                trend_data = []
                for stat in daily_stats:
                    trend_data.append({
                        'date': stat['date'],
                        'jobs': stat['total'],
                        'completed': stat['completed'],
                        'failed': stat['failed'],
                        'success_rate': stat['success_rate']
                    })
                
                return {
                    'period': f"Last {days} days",
                    'group_by': group_by.value,
                    'total_jobs': job_counts.get('total', 0),
                    'completed_jobs': job_counts.get('completed', 0),
                    'failed_jobs': job_counts.get('failed', 0),
                    'success_rate': self._calculate_success_rate_from_counts(job_counts),
                    'trend_data': trend_data,
                    'timestamp': datetime.utcnow().isoformat()
                }
            
            # 返回模拟趋势数据
            return self._generate_simulated_trend_data(days)
            
        except Exception as e:
            logger.error(f"Failed to get trend statistics: {e}")
            raise BusinessLogicError(
                f"获取趋势统计信息失败: {e}",
                operation="get_trend_statistics"
            )
    
    def _generate_simulated_trend_data(self, days: int) -> Dict[str, Any]:
        """生成模拟趋势数据"""
        trend_data = []
        for i in range(days):
            date = (datetime.utcnow() - timedelta(days=days-i-1)).strftime("%Y-%m-%d")
            trend_data.append({
                'date': date,
                'jobs': 0,
                'completed': 0,
                'failed': 0,
                'success_rate': 0
            })
        
        return {
            'period': f"Last {days} days",
            'group_by': 'day',
            'total_jobs': 0,
            'completed_jobs': 0,
            'failed_jobs': 0,
            'success_rate': 0,
            'trend_data': trend_data,
            'timestamp': datetime.utcnow().isoformat()
        }
    
    # ==================== 资源统计 ====================
    
    def get_resource_statistics(
        self,
        tenant_id: str = None,
        user_id: str = None,
        days: int = 7
    ) -> Dict[str, Any]:
        """获取资源使用统计
        
        Args:
            tenant_id: 租户ID
            user_id: 用户ID
            days: 统计天数
            
        Returns:
            资源使用统计
        """
        try:
            if self._repository and tenant_id:
                end_date = datetime.utcnow()
                start_date = end_date - timedelta(days=days)
                
                resource_stats = self._repository.get_resource_usage_statistics(
                    tenant_id=tenant_id,
                    user_id=user_id,
                    start_date=start_date,
                    end_date=end_date
                )
                
                return {
                    'period': f"Last {days} days",
                    'gpu': {
                        'average_utilization': round(resource_stats['gpu']['average_utilization'], 2),
                        'max_utilization': round(resource_stats['gpu']['max_utilization'], 2),
                        'average_memory_used_gb': round(resource_stats['gpu']['average_memory_used_gb'], 2)
                    },
                    'cpu': {
                        'average_utilization': round(resource_stats['cpu']['average_utilization'], 2),
                        'max_utilization': round(resource_stats['cpu']['max_utilization'], 2),
                        'average_memory_used_gb': round(resource_stats['cpu']['average_memory_used_gb'], 2)
                    },
                    'timestamp': datetime.utcnow().isoformat()
                }
            
            return {
                'period': f"Last {days} days",
                'gpu': {'average_utilization': 0, 'max_utilization': 0, 'average_memory_used_gb': 0},
                'cpu': {'average_utilization': 0, 'max_utilization': 0, 'average_memory_used_gb': 0},
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get resource statistics: {e}")
            raise BusinessLogicError(
                f"获取资源统计失败: {e}",
                operation="get_resource_statistics"
            )
    
    # ==================== 模型统计 ====================
    
    def get_model_statistics(
        self,
        tenant_id: str = None,
        user_id: str = None,
        limit: int = 10
    ) -> Dict[str, Any]:
        """获取模型使用统计
        
        Args:
            tenant_id: 租户ID
            user_id: 用户ID
            limit: 返回数量
            
        Returns:
            模型使用统计
        """
        try:
            if self._repository and tenant_id:
                model_stats = self._repository.get_model_usage_statistics(
                    tenant_id=tenant_id,
                    user_id=user_id,
                    limit=limit
                )
                
                return {
                    'top_models': model_stats,
                    'total_models': len(model_stats),
                    'timestamp': datetime.utcnow().isoformat()
                }
            
            return {
                'top_models': [],
                'total_models': 0,
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get model statistics: {e}")
            raise BusinessLogicError(
                f"获取模型统计失败: {e}",
                operation="get_model_statistics"
            )
    
    # ==================== 场景统计 ====================
    
    def get_scenario_statistics(
        self,
        tenant_id: str = None,
        user_id: str = None
    ) -> Dict[str, Any]:
        """获取训练场景统计
        
        Args:
            tenant_id: 租户ID
            user_id: 用户ID
            
        Returns:
            场景统计
        """
        try:
            if self._repository and tenant_id:
                scenario_stats = self._repository.get_scenario_statistics(
                    tenant_id=tenant_id,
                    user_id=user_id
                )
                
                return {
                    'scenarios': scenario_stats,
                    'total_scenarios': len(scenario_stats),
                    'timestamp': datetime.utcnow().isoformat()
                }
            
            return {
                'scenarios': [],
                'total_scenarios': 0,
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get scenario statistics: {e}")
            raise BusinessLogicError(
                f"获取场景统计失败: {e}",
                operation="get_scenario_statistics"
            )
    
    # ==================== 性能指标统计 ====================
    
    def get_performance_statistics(
        self,
        tenant_id: str = None,
        user_id: str = None
    ) -> Dict[str, Any]:
        """获取性能指标统计
        
        Args:
            tenant_id: 租户ID
            user_id: 用户ID
            
        Returns:
            性能指标统计
        """
        try:
            if self._repository and tenant_id:
                perf_stats = self._repository.get_performance_metrics_statistics(
                    tenant_id=tenant_id,
                    user_id=user_id
                )
                
                return {
                    'loss': perf_stats.get('loss', {}),
                    'accuracy': perf_stats.get('accuracy', {}),
                    'throughput': perf_stats.get('throughput', {}),
                    'timestamp': datetime.utcnow().isoformat()
                }
            
            return {
                'loss': {'average': 0, 'minimum': 0},
                'accuracy': {'average': 0, 'maximum': 0},
                'throughput': {'average_samples_per_second': 0, 'max_samples_per_second': 0},
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get performance statistics: {e}")
            raise BusinessLogicError(
                f"获取性能指标统计失败: {e}",
                operation="get_performance_statistics"
            )
    
    # ==================== 任务统计 ====================
    
    def get_job_statistics(
        self,
        tenant_id: str = None,
        user_id: str = None,
        job_id: str = None
    ) -> Dict[str, Any]:
        """获取特定任务的统计信息
        
        Args:
            tenant_id: 租户ID
            user_id: 用户ID
            job_id: 任务ID
            
        Returns:
            任务统计信息
        """
        try:
            if self._scenario_manager and job_id:
                job_status = self._scenario_manager.get_job_status(job_id)
                
                if not job_status:
                    raise ValidationError(f"任务未找到: {job_id}")
                
                return {
                    'job_id': job_id,
                    'status': job_status.get('status'),
                    'created_at': job_status.get('created_at'),
                    'result': job_status.get('result'),
                    'error': job_status.get('error'),
                    'duration': self._calculate_duration(job_status.get('created_at')),
                    'timestamp': datetime.utcnow().isoformat()
                }
            
            return {
                'job_id': job_id,
                'status': 'unknown',
                'message': 'Job not found or service unavailable',
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except ValidationError:
            raise
        except Exception as e:
            logger.error(f"Failed to get job statistics: {e}")
            raise BusinessLogicError(
                f"获取任务统计信息失败: {e}",
                operation="get_job_statistics"
            )
    
    # ==================== 整体统计 ====================
    
    def get_overall_statistics(
        self,
        tenant_id: str = None,
        user_id: str = None
    ) -> Dict[str, Any]:
        """获取整体训练统计信息
        
        Args:
            tenant_id: 租户ID
            user_id: 用户ID
            
        Returns:
            整体统计信息
        """
        try:
            if self._repository and tenant_id:
                comprehensive = self._repository.get_comprehensive_statistics(
                    tenant_id=tenant_id,
                    user_id=user_id,
                    days=30
                )
                
                return {
                    **comprehensive,
                    'uptime': self._get_uptime()
                }
            
            # 尝试从场景管理器获取
            if self._scenario_manager:
                try:
                    stats = self._scenario_manager.get_statistics()
                    return {
                        **stats,
                        'uptime': self._get_uptime(),
                        'timestamp': datetime.utcnow().isoformat()
                    }
                except Exception as e:
                    logger.warning(f"Failed to get stats from scenario manager: {e}")
            
            return {
                'summary': self._get_default_basic_statistics(),
                'uptime': self._get_uptime(),
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get overall statistics: {e}")
            raise BusinessLogicError(
                f"获取整体训练统计信息失败: {e}",
                operation="get_overall_statistics"
            )
    
    # ==================== 实时统计 ====================
    
    def get_realtime_statistics(
        self,
        tenant_id: str = None,
        user_id: str = None
    ) -> Dict[str, Any]:
        """获取实时统计信息
        
        Args:
            tenant_id: 租户ID
            user_id: 用户ID
            
        Returns:
            实时统计信息
        """
        try:
            if self._repository and tenant_id:
                # 获取今日统计
                today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
                
                job_counts = self._repository.get_job_count_by_status(
                    tenant_id=tenant_id,
                    user_id=user_id,
                    start_date=today_start
                )
                
                return {
                    'today': {
                        'total_jobs': job_counts.get('total', 0),
                        'running_jobs': job_counts.get('running', 0),
                        'completed_jobs': job_counts.get('completed', 0),
                        'failed_jobs': job_counts.get('failed', 0)
                    },
                    'current_time': datetime.utcnow().isoformat(),
                    'uptime': self._get_uptime()
                }
            
            return {
                'today': {
                    'total_jobs': 0,
                    'running_jobs': 0,
                    'completed_jobs': 0,
                    'failed_jobs': 0
                },
                'current_time': datetime.utcnow().isoformat(),
                'uptime': self._get_uptime()
            }
            
        except Exception as e:
            logger.error(f"Failed to get realtime statistics: {e}")
            raise BusinessLogicError(
                f"获取实时统计失败: {e}",
                operation="get_realtime_statistics"
            )
    
    # ==================== 辅助方法 ====================
    
    def _calculate_success_rate(self, stats: Dict[str, Any]) -> float:
        """计算成功率"""
        total = stats.get('total_jobs', 0)
        completed = stats.get('completed_jobs', 0)
        
        if total == 0:
            return 0.0
        
        return round(completed / total, 4)
    
    def _calculate_success_rate_from_counts(self, counts: Dict[str, int]) -> float:
        """从计数统计计算成功率"""
        total = counts.get('total', 0)
        completed = counts.get('completed', 0)
        
        if total == 0:
            return 0.0
        
        return round(completed / total * 100, 2)
    
    def _calculate_duration(self, start_time_str: str) -> str:
        """计算任务持续时间"""
        if not start_time_str:
            return "0 seconds"
        
        try:
            if isinstance(start_time_str, str):
                start_time = datetime.fromisoformat(start_time_str.replace('Z', '+00:00'))
            else:
                start_time = start_time_str
            
            duration = datetime.utcnow() - start_time
            seconds = int(duration.total_seconds())
            
            if seconds < 60:
                return f"{seconds} seconds"
            elif seconds < 3600:
                return f"{seconds // 60} minutes"
            else:
                hours = seconds // 3600
                minutes = (seconds % 3600) // 60
                return f"{hours} hours {minutes} minutes"
                
        except Exception:
            return "0 seconds"
    
    def _get_uptime(self) -> str:
        """获取服务运行时间"""
        try:
            uptime = datetime.utcnow() - self._service_start_time
            days = uptime.days
            hours = uptime.seconds // 3600
            minutes = (uptime.seconds % 3600) // 60
            
            parts = []
            if days > 0:
                parts.append(f"{days} days")
            if hours > 0:
                parts.append(f"{hours} hours")
            if minutes > 0 or not parts:
                parts.append(f"{minutes} minutes")
            
            return ", ".join(parts)
            
        except Exception:
            return "0 minutes"
    
    # ==================== 导出功能 ====================
    
    def export_statistics(
        self,
        tenant_id: str,
        user_id: str = None,
        format: str = 'json',
        days: int = 30
    ) -> Dict[str, Any]:
        """导出统计数据
        
        Args:
            tenant_id: 租户ID
            user_id: 用户ID
            format: 导出格式 (json/csv)
            days: 统计天数
            
        Returns:
            导出的统计数据
        """
        try:
            comprehensive = self.get_overall_statistics(tenant_id, user_id)
            detailed = self.get_detailed_statistics(tenant_id, user_id, days)
            trends = self.get_trend_statistics(tenant_id, user_id, days)
            
            export_data = {
                'metadata': {
                    'tenant_id': tenant_id,
                    'user_id': user_id,
                    'export_time': datetime.utcnow().isoformat(),
                    'period_days': days,
                    'format': format
                },
                'overall': comprehensive,
                'detailed': detailed,
                'trends': trends
            }
            
            return export_data
            
        except Exception as e:
            logger.error(f"Failed to export statistics: {e}")
            raise BusinessLogicError(
                f"导出统计数据失败: {e}",
                operation="export_statistics"
            )
    
    # ==================== Launcher 集成方法 ====================
    
    def get_launcher_statistics(self) -> Dict[str, Any]:
        """获取启动器相关统计信息
        
        Returns:
            启动器统计信息
        """
        try:
            # 获取模块可用性
            availability = get_module_availability()
            
            # 获取诊断信息
            diagnostics = diagnose_launcher_module()
            
            # 获取所有训练模式
            training_modes = get_all_training_modes()
            
            return {
                'available': True,
                'module_availability': availability,
                'diagnostics': diagnostics,
                'training_modes': training_modes,
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get launcher statistics: {e}")
            return {
                'available': False,
                'error': str(e)
            }
    
    def get_progress_statistics(
        self,
        tenant_id: str = None,
        user_id: str = None
    ) -> Dict[str, Any]:
        """获取训练进度统计信息
        
        Args:
            tenant_id: 租户ID
            user_id: 用户ID
            
        Returns:
            进度统计信息
        """
        if self._progress_manager is None:
            return {
                'available': False,
                'error': 'Progress manager not available'
            }
        
        try:
            # 获取所有活跃的训练进度
            all_progress = self._progress_manager.get_all_progress()
            
            active_trainings = []
            for session_id, progress in all_progress.items():
                active_trainings.append({
                    'session_id': session_id,
                    'status': progress.status,
                    'progress': progress.progress,
                    'current_step': progress.current_step,
                    'total_steps': progress.total_steps,
                    'current_epoch': progress.current_epoch,
                    'current_stage': progress.current_stage,
                    'start_time': progress.start_time.isoformat() if progress.start_time else None
                })
            
            return {
                'available': True,
                'active_trainings': active_trainings,
                'active_count': len(active_trainings),
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get progress statistics: {e}")
            return {
                'available': False,
                'error': str(e)
            }
    
    def get_system_health_statistics(self) -> Dict[str, Any]:
        """获取系统健康统计信息
        
        Returns:
            系统健康统计
        """
        health_info = {
            'repository_available': self._repository is not None,
            'scenario_manager_available': self._scenario_manager is not None,
            'uptime': self._get_uptime(),
            'timestamp': datetime.utcnow().isoformat()
        }
        
        # 添加启动器模块可用性
        try:
            health_info['module_availability'] = get_module_availability()
        except Exception as e:
            health_info['module_availability_error'] = str(e)
        
        # 添加进度管理器状态
        if self._progress_manager:
            try:
                all_progress = self._progress_manager.get_all_progress()
                health_info['active_trainings_count'] = len(all_progress)
            except Exception as e:
                health_info['progress_error'] = str(e)
        
        return health_info


# ==================== 全局实例管理 ====================

_global_training_statistics_service: Optional[TrainingStatisticsService] = None


def get_training_statistics_service(
    use_memory_storage: bool = False
) -> TrainingStatisticsService:
    """获取全局训练统计服务实例
    
    Args:
        use_memory_storage: 是否使用内存存储
        
    Returns:
        训练统计服务实例
    """
    global _global_training_statistics_service
    
    if _global_training_statistics_service is None:
        _global_training_statistics_service = TrainingStatisticsService(
            use_memory_storage=use_memory_storage
        )
    
    return _global_training_statistics_service


def reset_training_statistics_service() -> None:
    """重置全局服务实例（用于测试）"""
    global _global_training_statistics_service
    _global_training_statistics_service = None

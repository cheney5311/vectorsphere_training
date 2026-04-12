"""用户训练服务

提供用户训练相关的业务逻辑处理，包括训练概览、会话管理、统计分析等功能。

服务层负责：
1. 组合数据仓库层的查询结果
2. 实现复杂的业务逻辑
3. 处理数据聚合和转换
4. 提供统一的服务接口
"""

import logging
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum

logger = logging.getLogger(__name__)


# =============================================================================
# 数据传输对象定义
# =============================================================================

@dataclass
class UserTrainingOverview:
    """用户训练概览数据
    
    Attributes:
        active_sessions: 活跃训练会话数
        completed_sessions: 已完成训练会话数
        total_models: 已训练的模型总数
        avg_accuracy: 平均准确率（百分比）
        success_rate: 成功率（百分比）
        total_training_hours: 总训练时长（小时）
    """
    active_sessions: int = 0
    completed_sessions: int = 0
    total_models: int = 0
    avg_accuracy: float = 0.0
    success_rate: float = 0.0
    total_training_hours: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典（驼峰命名）"""
        return {
            'activeSessions': self.active_sessions,
            'completedSessions': self.completed_sessions,
            'totalModels': self.total_models,
            'avgAccuracy': round(self.avg_accuracy, 2),
            'successRate': round(self.success_rate, 2),
            'totalTrainingHours': round(self.total_training_hours, 2)
        }


@dataclass
class TrainingSessionInfo:
    """训练会话信息
    
    Attributes:
        id: 会话ID
        name: 会话名称
        model_type: 模型类型
        status: 状态
        progress: 进度（0-100）
        accuracy: 准确率
        loss: 损失值
        start_time: 开始时间
        end_time: 结束时间
        duration_minutes: 持续时长（分钟）
        current_epoch: 当前轮次
        total_epochs: 总轮次
        error_message: 错误信息
    """
    id: str = ""
    name: str = ""
    model_type: str = ""
    status: str = ""
    progress: float = 0.0
    accuracy: float = 0.0
    loss: float = 0.0
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    duration_minutes: float = 0.0
    current_epoch: int = 0
    total_epochs: int = 0
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'id': self.id,
            'name': self.name,
            'modelType': self.model_type,
            'status': self.status,
            'progress': round(self.progress, 1),
            'accuracy': round(self.accuracy, 4),
            'loss': round(self.loss, 6) if self.loss else 0,
            'startTime': self.start_time,
            'endTime': self.end_time,
            'durationMinutes': round(self.duration_minutes, 1),
            'currentEpoch': self.current_epoch,
            'totalEpochs': self.total_epochs,
            'errorMessage': self.error_message
        }


@dataclass
class UserTrainingStatistics:
    """用户训练统计
    
    Attributes:
        total_tasks: 总任务数
        completed_tasks: 已完成任务数
        running_tasks: 运行中任务数
        pending_tasks: 等待中任务数
        failed_tasks: 失败任务数
        cancelled_tasks: 已取消任务数
        success_rate: 成功率（百分比）
        avg_training_time: 平均训练时间（分钟）
        total_training_time: 总训练时间（小时）
        avg_accuracy: 平均准确率
        best_accuracy: 最佳准确率
        avg_loss: 平均损失
        best_loss: 最低损失
    """
    total_tasks: int = 0
    completed_tasks: int = 0
    running_tasks: int = 0
    pending_tasks: int = 0
    failed_tasks: int = 0
    cancelled_tasks: int = 0
    success_rate: float = 0.0
    avg_training_time: float = 0.0
    total_training_time: float = 0.0
    avg_accuracy: float = 0.0
    best_accuracy: float = 0.0
    avg_loss: float = 0.0
    best_loss: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典（驼峰命名）"""
        return {
            'totalTasks': self.total_tasks,
            'completedTasks': self.completed_tasks,
            'runningTasks': self.running_tasks,
            'pendingTasks': self.pending_tasks,
            'failedTasks': self.failed_tasks,
            'cancelledTasks': self.cancelled_tasks,
            'successRate': round(self.success_rate, 2),
            'avgTrainingTime': round(self.avg_training_time, 2),
            'totalTrainingTime': round(self.total_training_time, 2),
            'avgAccuracy': round(self.avg_accuracy, 4),
            'bestAccuracy': round(self.best_accuracy, 4),
            'avgLoss': round(self.avg_loss, 6),
            'bestLoss': round(self.best_loss, 6)
        }


@dataclass
class TrainingTrendData:
    """训练趋势数据
    
    Attributes:
        date: 日期 (YYYY-MM-DD)
        completed: 完成数
        running: 运行中数
        failed: 失败数
        total: 总数
        avg_accuracy: 当日平均准确率
    """
    date: str = ""
    completed: int = 0
    running: int = 0
    failed: int = 0
    total: int = 0
    avg_accuracy: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ModelPerformance:
    """模型性能数据
    
    Attributes:
        model_id: 模型ID
        model_name: 模型名称
        model_type: 模型类型
        best_accuracy: 最佳准确率
        best_loss: 最低损失
        training_count: 训练次数
        last_trained: 最后训练时间
    """
    model_id: str = ""
    model_name: str = ""
    model_type: str = ""
    best_accuracy: float = 0.0
    best_loss: float = 0.0
    training_count: int = 0
    last_trained: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'modelId': self.model_id,
            'modelName': self.model_name,
            'modelType': self.model_type,
            'bestAccuracy': round(self.best_accuracy, 4),
            'bestLoss': round(self.best_loss, 6),
            'trainingCount': self.training_count,
            'lastTrained': self.last_trained
        }


# =============================================================================
# 用户训练服务
# =============================================================================

class UserTrainingService:
    """用户训练服务
    
    提供用户训练相关的业务逻辑处理。
    
    主要功能:
        - 获取用户训练概览
        - 获取用户训练会话列表
        - 获取用户训练统计
        - 获取训练趋势数据
        - 获取模型性能排行
    
    Example:
        >>> service = get_user_training_service()
        >>> overview = service.get_user_overview("user123")
        >>> print(f"活跃会话: {overview.active_sessions}")
    """
    
    def __init__(self, repository=None):
        """初始化用户训练服务
        
        Args:
            repository: 用户训练仓库实例，如果为None则自动创建
        """
        if repository is None:
            from backend.repositories.user_training_repository import get_user_training_repository
            repository = get_user_training_repository()
        self.repository = repository
    
    # =========================================================================
    # 概览相关
    # =========================================================================
    
    def get_user_overview(
        self,
        user_id: str,
        tenant_id: Optional[str] = None
    ) -> UserTrainingOverview:
        """获取用户训练概览
        
        汇总用户的训练活动数据，包括活跃会话、完成会话、模型数量和平均准确率。
        
        Args:
            user_id: 用户ID
            tenant_id: 租户ID（可选）
            
        Returns:
            UserTrainingOverview: 用户训练概览数据
            
        Example:
            >>> overview = service.get_user_overview("user123")
            >>> print(overview.to_dict())
            {'activeSessions': 2, 'completedSessions': 15, ...}
        """
        try:
            overview_data = self.repository.get_user_training_overview(
                user_id=user_id,
                tenant_id=tenant_id
            )
            
            return UserTrainingOverview(
                active_sessions=overview_data.get('active_count', 0),
                completed_sessions=overview_data.get('completed_count', 0),
                total_models=overview_data.get('total_models', 0),
                avg_accuracy=overview_data.get('avg_accuracy', 0.0),
                success_rate=overview_data.get('success_rate', 0.0),
                total_training_hours=overview_data.get('total_training_hours', 0.0)
            )
            
        except Exception as e:
            logger.error(f"Failed to get user overview for {user_id}: {e}")
            return UserTrainingOverview()
    
    # =========================================================================
    # 会话相关
    # =========================================================================
    
    def get_recent_sessions(
        self,
        user_id: str,
        limit: int = 5,
        tenant_id: Optional[str] = None
    ) -> List[TrainingSessionInfo]:
        """获取用户最近的训练会话
        
        Args:
            user_id: 用户ID
            limit: 返回数量限制，默认5，最大20
            tenant_id: 租户ID（可选）
            
        Returns:
            List[TrainingSessionInfo]: 最近训练会话列表
            
        Example:
            >>> sessions = service.get_recent_sessions("user123", limit=10)
            >>> for s in sessions:
            ...     print(f"{s.name}: {s.status}")
        """
        try:
            limit = min(max(limit, 1), 20)
            
            sessions = self.repository.get_recent_sessions(
                user_id=user_id,
                limit=limit,
                tenant_id=tenant_id
            )
            
            return [self._convert_session_to_info(s) for s in sessions]
            
        except Exception as e:
            logger.error(f"Failed to get recent sessions for {user_id}: {e}")
            return []
    
    def get_user_sessions(
        self,
        user_id: str,
        page: int = 1,
        limit: int = 10,
        status: Optional[str] = None,
        model_type: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        sort_by: str = 'created_at',
        sort_order: str = 'desc',
        tenant_id: Optional[str] = None
    ) -> Tuple[List[TrainingSessionInfo], int]:
        """获取用户训练会话列表（支持分页和筛选）
        
        Args:
            user_id: 用户ID
            page: 页码，从1开始
            limit: 每页数量，默认10，最大50
            status: 状态筛选（可选）
            model_type: 模型类型筛选（可选）
            start_date: 开始日期筛选（可选）
            end_date: 结束日期筛选（可选）
            sort_by: 排序字段，默认 'created_at'
            sort_order: 排序方向，'asc' 或 'desc'
            tenant_id: 租户ID（可选）
            
        Returns:
            Tuple[List[TrainingSessionInfo], int]: (会话列表, 总数)
            
        Example:
            >>> sessions, total = service.get_user_sessions(
            ...     user_id="user123",
            ...     page=1,
            ...     limit=10,
            ...     status="completed"
            ... )
            >>> print(f"共 {total} 条，当前页 {len(sessions)} 条")
        """
        try:
            page = max(page, 1)
            limit = min(max(limit, 1), 50)
            offset = (page - 1) * limit
            
            sessions, total = self.repository.get_user_sessions(
                user_id=user_id,
                offset=offset,
                limit=limit,
                status=status,
                model_type=model_type,
                start_date=start_date,
                end_date=end_date,
                sort_by=sort_by,
                sort_order=sort_order,
                tenant_id=tenant_id
            )
            
            session_list = [self._convert_session_to_info(s) for s in sessions]
            
            return session_list, total
            
        except Exception as e:
            logger.error(f"Failed to get user sessions for {user_id}: {e}")
            return [], 0
    
    def get_session_detail(
        self,
        session_id: str,
        user_id: str,
        tenant_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """获取训练会话详情
        
        Args:
            session_id: 会话ID
            user_id: 用户ID（用于权限验证）
            tenant_id: 租户ID（可选）
            
        Returns:
            Dict: 会话详情，如果不存在或无权限则返回None
            
        Example:
            >>> detail = service.get_session_detail("sess_123", "user123")
            >>> if detail:
            ...     print(f"会话配置: {detail['config']}")
        """
        try:
            return self.repository.get_session_detail(
                session_id=session_id,
                user_id=user_id,
                tenant_id=tenant_id
            )
        except Exception as e:
            logger.error(f"Failed to get session detail {session_id}: {e}")
            return None
    
    def get_active_sessions(
        self,
        user_id: str,
        tenant_id: Optional[str] = None
    ) -> List[TrainingSessionInfo]:
        """获取用户活跃训练会话
        
        获取状态为 running、pending、training 的会话。
        
        Args:
            user_id: 用户ID
            tenant_id: 租户ID（可选）
            
        Returns:
            List[TrainingSessionInfo]: 活跃会话列表
        """
        try:
            sessions = self.repository.get_active_sessions(
                user_id=user_id,
                tenant_id=tenant_id
            )
            return [self._convert_session_to_info(s) for s in sessions]
        except Exception as e:
            logger.error(f"Failed to get active sessions for {user_id}: {e}")
            return []
    
    # =========================================================================
    # 统计相关
    # =========================================================================
    
    def get_user_statistics(
        self,
        user_id: str,
        days: int = 30,
        tenant_id: Optional[str] = None
    ) -> UserTrainingStatistics:
        """获取用户训练统计
        
        汇总用户在指定时间范围内的训练统计数据。
        
        Args:
            user_id: 用户ID
            days: 统计天数，默认30天
            tenant_id: 租户ID（可选）
            
        Returns:
            UserTrainingStatistics: 用户训练统计数据
            
        Example:
            >>> stats = service.get_user_statistics("user123", days=30)
            >>> print(f"成功率: {stats.success_rate}%")
        """
        try:
            stats_data = self.repository.get_user_statistics(
                user_id=user_id,
                days=days,
                tenant_id=tenant_id
            )
            
            return UserTrainingStatistics(
                total_tasks=stats_data.get('total_count', 0),
                completed_tasks=stats_data.get('completed_count', 0),
                running_tasks=stats_data.get('running_count', 0),
                pending_tasks=stats_data.get('pending_count', 0),
                failed_tasks=stats_data.get('failed_count', 0),
                cancelled_tasks=stats_data.get('cancelled_count', 0),
                success_rate=stats_data.get('success_rate', 0.0),
                avg_training_time=stats_data.get('avg_training_time', 0.0),
                total_training_time=stats_data.get('total_training_time', 0.0),
                avg_accuracy=stats_data.get('avg_accuracy', 0.0),
                best_accuracy=stats_data.get('best_accuracy', 0.0),
                avg_loss=stats_data.get('avg_loss', 0.0),
                best_loss=stats_data.get('best_loss', 0.0)
            )
            
        except Exception as e:
            logger.error(f"Failed to get user statistics for {user_id}: {e}")
            return UserTrainingStatistics()
    
    def get_training_trend(
        self,
        user_id: str,
        days: int = 7,
        tenant_id: Optional[str] = None
    ) -> List[TrainingTrendData]:
        """获取训练趋势数据
        
        按日期统计训练任务的数量变化趋势。
        
        Args:
            user_id: 用户ID
            days: 统计天数，默认7天，最大90天
            tenant_id: 租户ID（可选）
            
        Returns:
            List[TrainingTrendData]: 每日趋势数据列表
            
        Example:
            >>> trend = service.get_training_trend("user123", days=14)
            >>> for day in trend:
            ...     print(f"{day.date}: 完成 {day.completed}, 失败 {day.failed}")
        """
        try:
            days = min(max(days, 1), 90)
            
            trend_data = self.repository.get_training_trend(
                user_id=user_id,
                days=days,
                tenant_id=tenant_id
            )
            
            return [
                TrainingTrendData(
                    date=d['date'],
                    completed=d.get('completed', 0),
                    running=d.get('running', 0),
                    failed=d.get('failed', 0),
                    total=d.get('total', 0),
                    avg_accuracy=d.get('avg_accuracy', 0.0)
                )
                for d in trend_data
            ]
            
        except Exception as e:
            logger.error(f"Failed to get training trend for {user_id}: {e}")
            return []
    
    def get_model_performance_ranking(
        self,
        user_id: str,
        limit: int = 10,
        tenant_id: Optional[str] = None
    ) -> List[ModelPerformance]:
        """获取模型性能排行
        
        按最佳准确率排序获取用户的模型性能排行。
        
        Args:
            user_id: 用户ID
            limit: 返回数量，默认10
            tenant_id: 租户ID（可选）
            
        Returns:
            List[ModelPerformance]: 模型性能列表（按准确率降序）
            
        Example:
            >>> ranking = service.get_model_performance_ranking("user123", limit=5)
            >>> for model in ranking:
            ...     print(f"{model.model_name}: {model.best_accuracy:.2%}")
        """
        try:
            models = self.repository.get_model_performance_ranking(
                user_id=user_id,
                limit=limit,
                tenant_id=tenant_id
            )
            
            return [
                ModelPerformance(
                    model_id=m.get('model_id', ''),
                    model_name=m.get('model_name', ''),
                    model_type=m.get('model_type', ''),
                    best_accuracy=m.get('best_accuracy', 0.0),
                    best_loss=m.get('best_loss', 0.0),
                    training_count=m.get('training_count', 0),
                    last_trained=m.get('last_trained')
                )
                for m in models
            ]
            
        except Exception as e:
            logger.error(f"Failed to get model performance ranking for {user_id}: {e}")
            return []
    
    # =========================================================================
    # 训练时长相关
    # =========================================================================
    
    def get_training_duration_stats(
        self,
        user_id: str,
        days: int = 30,
        tenant_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """获取训练时长统计
        
        统计用户训练任务的时长分布。
        
        Args:
            user_id: 用户ID
            days: 统计天数
            tenant_id: 租户ID（可选）
            
        Returns:
            Dict: 时长统计数据，包含:
                - avgDuration: 平均时长（分钟）
                - minDuration: 最短时长（分钟）
                - maxDuration: 最长时长（分钟）
                - totalDuration: 总时长（小时）
                - distribution: 时长分布
        """
        try:
            return self.repository.get_training_duration_stats(
                user_id=user_id,
                days=days,
                tenant_id=tenant_id
            )
        except Exception as e:
            logger.error(f"Failed to get training duration stats for {user_id}: {e}")
            return {
                'avgDuration': 0,
                'minDuration': 0,
                'maxDuration': 0,
                'totalDuration': 0,
                'distribution': []
            }
    
    # =========================================================================
    # 辅助方法
    # =========================================================================
    
    def _convert_session_to_info(self, session: Dict[str, Any]) -> TrainingSessionInfo:
        """将会话数据转换为 TrainingSessionInfo
        
        Args:
            session: 会话数据字典
            
        Returns:
            TrainingSessionInfo: 转换后的会话信息
        """
        # 计算持续时长
        duration_minutes = 0.0
        start_time = session.get('started_at') or session.get('created_at')
        end_time = session.get('completed_at')
        
        if start_time and end_time:
            if isinstance(start_time, str):
                start_time = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
            if isinstance(end_time, str):
                end_time = datetime.fromisoformat(end_time.replace('Z', '+00:00'))
            duration_minutes = (end_time - start_time).total_seconds() / 60
        elif start_time and session.get('status') in ['running', 'training']:
            if isinstance(start_time, str):
                start_time = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
            duration_minutes = (datetime.utcnow() - start_time).total_seconds() / 60
        
        # 格式化时间
        start_time_str = None
        end_time_str = None
        if session.get('started_at'):
            st = session['started_at']
            start_time_str = st.isoformat() if hasattr(st, 'isoformat') else str(st)
        elif session.get('created_at'):
            ct = session['created_at']
            start_time_str = ct.isoformat() if hasattr(ct, 'isoformat') else str(ct)
        
        if session.get('completed_at'):
            et = session['completed_at']
            end_time_str = et.isoformat() if hasattr(et, 'isoformat') else str(et)
        
        # 获取名称
        name = session.get('name') or session.get('session_name')
        if not name and session.get('config'):
            name = session['config'].get('session_name')
        if not name:
            session_id = session.get('session_id') or session.get('id', '')
            name = f"Training {session_id[:8]}" if session_id else "Unknown"
        
        return TrainingSessionInfo(
            id=session.get('session_id') or session.get('id', ''),
            name=name,
            model_type=session.get('training_type') or session.get('model_type', 'unknown'),
            status=session.get('status', 'unknown'),
            progress=float(session.get('progress', 0) or 0),
            accuracy=float(session.get('accuracy', 0) or 0),
            loss=float(session.get('loss', 0) or 0),
            start_time=start_time_str,
            end_time=end_time_str,
            duration_minutes=duration_minutes,
            current_epoch=session.get('current_epoch', 0) or 0,
            total_epochs=session.get('total_epochs', 0) or 0,
            error_message=session.get('error_message')
        )


# =============================================================================
# 全局服务实例
# =============================================================================

_user_training_service: Optional[UserTrainingService] = None


def get_user_training_service() -> UserTrainingService:
    """获取用户训练服务单例
    
    Returns:
        UserTrainingService: 用户训练服务实例
        
    Example:
        >>> service = get_user_training_service()
        >>> overview = service.get_user_overview("user123")
    """
    global _user_training_service
    if _user_training_service is None:
        _user_training_service = UserTrainingService()
    return _user_training_service


def reset_user_training_service():
    """重置用户训练服务（用于测试）"""
    global _user_training_service
    _user_training_service = None

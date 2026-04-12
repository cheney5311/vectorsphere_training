"""仪表盘数据仓库

聚合多数据源的查询，为仪表盘提供统一的数据访问接口。
从训练、模型、系统监控等多个数据源获取数据并汇总。
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from sqlalchemy import func, and_, desc
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError

logger = logging.getLogger(__name__)


class DashboardRepository:
    """仪表盘数据仓库
    
    聚合训练、模型、数据集、系统监控等多数据源的查询。
    
    Attributes:
        db_manager: 数据库管理器实例
    """
    
    def __init__(self, db_manager=None):
        """初始化仪表盘仓库
        
        Args:
            db_manager: 数据库管理器，为 None 时自动获取
        """
        self._db_manager = db_manager
        
    def _get_db_manager(self):
        """延迟获取数据库管理器"""
        if self._db_manager is None:
            from backend.modules.database.manager import get_database_manager
            self._db_manager = get_database_manager()
        return self._db_manager
    
    def _get_session(self) -> Session:
        """获取数据库会话"""
        return self._get_db_manager().get_session()
    
    # =========================================================================
    # 训练统计查询
    # =========================================================================
    
    def get_training_overview(
        self, 
        user_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """获取训练概览数据
        
        Args:
            user_id: 用户ID（可选，用于筛选）
            tenant_id: 租户ID（可选，用于筛选）
            start_date: 开始时间
            end_date: 结束时间
            
        Returns:
            Dict: 训练概览数据，包含各状态的任务数量
        """
        session = self._get_session()
        try:
            from backend.schemas.training_models import TrainingSession
            
            # 构建查询条件
            conditions = []
            if user_id:
                conditions.append(TrainingSession.user_id == user_id)
            if tenant_id:
                conditions.append(TrainingSession.tenant_id == tenant_id)
            if start_date:
                conditions.append(TrainingSession.created_at >= start_date)
            if end_date:
                conditions.append(TrainingSession.created_at <= end_date)
            
            # 查询各状态数量
            # pylint: disable=not-callable
            query = session.query(
                TrainingSession.status,
                func.count(TrainingSession.session_id).label('count')
            )
            
            if conditions:
                query = query.filter(and_(*conditions))
            
            status_counts = query.group_by(TrainingSession.status).all()
            
            # 整理结果
            result = {
                'active_count': 0,
                'completed_count': 0,
                'failed_count': 0,
                'pending_count': 0,
                'paused_count': 0,
                'total_count': 0
            }
            
            status_mapping = {
                'running': 'active_count',
                'completed': 'completed_count',
                'failed': 'failed_count',
                'pending': 'pending_count',
                'paused': 'paused_count',
                'stopped': 'completed_count'  # 停止也算完成
            }
            
            for status, count in status_counts:
                key = status_mapping.get(status)
                if key:
                    result[key] += count
                result['total_count'] += count
            
            # 计算成功率
            total_finished = result['completed_count'] + result['failed_count']
            result['success_rate'] = (
                result['completed_count'] / total_finished 
                if total_finished > 0 else 0.0
            )
            
            # 计算训练时长统计
            # pylint: disable=not-callable
            time_query = session.query(
                func.sum(
                    func.extract('epoch', TrainingSession.completed_at) -
                    func.extract('epoch', TrainingSession.started_at)
                ).label('total_seconds'),
                func.count(TrainingSession.session_id).label('count')
            ).filter(
                TrainingSession.started_at.isnot(None),
                TrainingSession.completed_at.isnot(None)
            )
            
            if conditions:
                time_query = time_query.filter(and_(*conditions))
            
            time_result = time_query.first()
            
            if time_result and time_result.total_seconds and time_result.count:
                total_hours = time_result.total_seconds / 3600
                result['total_training_time_hours'] = total_hours
                result['avg_training_time_hours'] = total_hours / time_result.count
            else:
                result['total_training_time_hours'] = 0.0
                result['avg_training_time_hours'] = 0.0
            
            return result
            
        except SQLAlchemyError as e:
            logger.error("Failed to get training overview: %s", e)
            return {
                'active_count': 0, 'completed_count': 0, 'failed_count': 0,
                'pending_count': 0, 'paused_count': 0, 'total_count': 0,
                'success_rate': 0.0, 'total_training_time_hours': 0.0,
                'avg_training_time_hours': 0.0
            }
        finally:
            session.close()
    
    def get_training_trends(
        self,
        user_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        # pylint: disable=unused-argument
        granularity: str = 'day'
    ) -> List[Dict[str, Any]]:
        """获取训练趋势数据
        
        Args:
            user_id: 用户ID
            tenant_id: 租户ID
            start_date: 开始时间
            end_date: 结束时间
            granularity: 粒度 (hour/day/week/month)
            
        Returns:
            List[Dict]: 趋势数据列表
        """
        session = self._get_session()
        try:
            from backend.schemas.training_models import TrainingSession
            
            # 设置默认时间范围
            if end_date is None:
                end_date = datetime.utcnow()
            if start_date is None:
                start_date = end_date - timedelta(days=7)
            
            # 构建查询条件
            conditions = [
                TrainingSession.created_at >= start_date,
                TrainingSession.created_at <= end_date
            ]
            if user_id:
                conditions.append(TrainingSession.user_id == user_id)
            if tenant_id:
                conditions.append(TrainingSession.tenant_id == tenant_id)
            
            # 根据粒度选择日期截断函数
            # 使用 SQLite/PostgreSQL 兼容的方式
            # pylint: disable=not-callable
            date_trunc_expr = func.date(TrainingSession.created_at)
            
            query = session.query(
                date_trunc_expr.label('date'),
                func.count(TrainingSession.session_id).label('count'),
                func.sum(
                    func.case(
                        (TrainingSession.status == 'completed', 1),
                        else_=0
                    )
                ).label('success_count'),
                func.sum(
                    func.case(
                        (TrainingSession.status == 'failed', 1),
                        else_=0
                    )
                ).label('failed_count')
            ).filter(
                and_(*conditions)
            ).group_by(
                date_trunc_expr
            ).order_by(
                date_trunc_expr
            )
            
            results = query.all()
            
            trends = []
            for row in results:
                date_val = row.date
                if isinstance(date_val, str):
                    date_val = datetime.fromisoformat(date_val)
                elif not isinstance(date_val, datetime):
                    # 处理 date 类型
                    date_val = datetime.combine(date_val, datetime.min.time())
                
                trends.append({
                    'timestamp': date_val.isoformat(),
                    'date_label': date_val.strftime('%Y-%m-%d'),
                    'count': row.count or 0,
                    'success_count': row.success_count or 0,
                    'failed_count': row.failed_count or 0,
                    'avg_duration_hours': 0.0  # 可以在后续计算
                })
            
            return trends
            
        except SQLAlchemyError as e:
            logger.error("Failed to get training trends: %s", e)
            return []
        finally:
            session.close()
    
    def get_training_by_type(
        self,
        user_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, int]:
        """获取按训练类型分组的统计
        
        Args:
            user_id: 用户ID
            tenant_id: 租户ID
            start_date: 开始时间
            end_date: 结束时间
            
        Returns:
            Dict[str, int]: 类型到数量的映射
        """
        session = self._get_session()
        try:
            from backend.schemas.training_models import TrainingSession
            
            conditions = []
            if user_id:
                conditions.append(TrainingSession.user_id == user_id)
            if tenant_id:
                conditions.append(TrainingSession.tenant_id == tenant_id)
            if start_date:
                conditions.append(TrainingSession.created_at >= start_date)
            if end_date:
                conditions.append(TrainingSession.created_at <= end_date)
            
            query = session.query(
                TrainingSession.training_type,
                # pylint: disable=not-callable
                func.count(TrainingSession.session_id).label('count')
            )
            
            if conditions:
                query = query.filter(and_(*conditions))
            
            results = query.group_by(TrainingSession.training_type).all()
            
            return {row.training_type or 'unknown': row.count for row in results}
            
        except SQLAlchemyError as e:
            logger.error("Failed to get training by type: %s", e)
            return {}
        finally:
            session.close()
    
    def get_recent_training_sessions(
        self,
        user_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """获取最近的训练会话
        
        Args:
            user_id: 用户ID
            tenant_id: 租户ID
            limit: 返回数量限制
            
        Returns:
            List[Dict]: 最近训练会话列表
        """
        session = self._get_session()
        try:
            from backend.schemas.training_models import TrainingSession
            
            conditions = []
            if user_id:
                conditions.append(TrainingSession.user_id == user_id)
            if tenant_id:
                conditions.append(TrainingSession.tenant_id == tenant_id)
            
            query = session.query(TrainingSession)
            
            if conditions:
                query = query.filter(and_(*conditions))
            
            sessions = query.order_by(
                desc(TrainingSession.created_at)
            ).limit(limit).all()
            
            return [s.to_dict() for s in sessions]
            
        except SQLAlchemyError as e:
            logger.error("Failed to get recent training sessions: %s", e)
            return []
        finally:
            session.close()
    
    # =========================================================================
    # 模型统计查询
    # =========================================================================
    
    def get_model_overview(
        self,
        user_id: Optional[str] = None,
        tenant_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """获取模型概览数据
        
        Args:
            user_id: 用户ID
            tenant_id: 租户ID
            
        Returns:
            Dict: 模型概览数据
        """
        session = self._get_session()
        try:
            from backend.schemas.model_models import ModelDB
            
            conditions = []
            if user_id:
                conditions.append(ModelDB.user_id == user_id)
            if tenant_id:
                conditions.append(ModelDB.tenant_id == tenant_id)
            
            # 基础统计
            base_query = session.query(ModelDB)
            if conditions:
                base_query = base_query.filter(and_(*conditions))
            
            total_count = base_query.count()
            
            # 按状态统计
            # pylint: disable=not-callable
            status_query = session.query(
                ModelDB.status,
                func.count(ModelDB.id).label('count')
            )
            if conditions:
                status_query = status_query.filter(and_(*conditions))
            
            status_results = status_query.group_by(ModelDB.status).all()
            
            status_counts = {row.status: row.count for row in status_results}
            
            # 准确率统计
            # pylint: disable=not-callable
            metrics_query = session.query(
                func.avg(ModelDB.accuracy).label('avg_accuracy'),
                func.max(ModelDB.accuracy).label('best_accuracy'),
                func.avg(ModelDB.f1_score).label('avg_f1_score'),
                func.sum(ModelDB.size_mb).label('total_size_mb')
            )
            if conditions:
                metrics_query = metrics_query.filter(and_(*conditions))
            
            metrics = metrics_query.first()
            
            return {
                'total_count': total_count,
                'deployed_count': status_counts.get('deployed', 0),
                'draft_count': status_counts.get('draft', 0),
                'archived_count': status_counts.get('archived', 0),
                'avg_accuracy': float(metrics.avg_accuracy or 0),
                'best_accuracy': float(metrics.best_accuracy or 0),
                'avg_f1_score': float(metrics.avg_f1_score or 0),
                'total_size_gb': float(metrics.total_size_mb or 0) / 1024
            }
            
        except SQLAlchemyError as e:
            logger.error("Failed to get model overview: %s", e)
            return {
                'total_count': 0, 'deployed_count': 0, 'draft_count': 0,
                'archived_count': 0, 'avg_accuracy': 0.0, 'best_accuracy': 0.0,
                'avg_f1_score': 0.0, 'total_size_gb': 0.0
            }
        finally:
            session.close()
    
    def get_model_distribution(
        self,
        user_id: Optional[str] = None,
        tenant_id: Optional[str] = None
    ) -> Dict[str, Dict[str, int]]:
        """获取模型分布数据
        
        Args:
            user_id: 用户ID
            tenant_id: 租户ID
            
        Returns:
            Dict: 按类型、框架、状态等分布的数据
        """
        session = self._get_session()
        try:
            from backend.schemas.model_models import ModelDB
            
            conditions = []
            if user_id:
                conditions.append(ModelDB.user_id == user_id)
            if tenant_id:
                conditions.append(ModelDB.tenant_id == tenant_id)
            
            result = {
                'by_type': {},
                'by_framework': {},
                'by_status': {},
                'by_category': {}
            }
            
            # 按类型
            # pylint: disable=not-callable
            type_query = session.query(
                ModelDB.model_type,
                func.count(ModelDB.id).label('count')
            )
            if conditions:
                type_query = type_query.filter(and_(*conditions))
            type_results = type_query.group_by(ModelDB.model_type).all()
            result['by_type'] = {row.model_type or 'unknown': row.count for row in type_results}
            
            # 按框架
            # pylint: disable=not-callable
            framework_query = session.query(
                ModelDB.framework,
                func.count(ModelDB.id).label('count')
            )
            if conditions:
                framework_query = framework_query.filter(and_(*conditions))
            framework_results = framework_query.group_by(ModelDB.framework).all()
            result['by_framework'] = {row.framework or 'unknown': row.count for row in framework_results}
            
            # 按状态
            # pylint: disable=not-callable
            status_query = session.query(
                ModelDB.status,
                func.count(ModelDB.id).label('count')
            )
            if conditions:
                status_query = status_query.filter(and_(*conditions))
            status_results = status_query.group_by(ModelDB.status).all()
            result['by_status'] = {row.status or 'unknown': row.count for row in status_results}
            
            # 按分类
            # pylint: disable=not-callable
            category_query = session.query(
                ModelDB.category,
                func.count(ModelDB.id).label('count')
            )
            if conditions:
                category_query = category_query.filter(and_(*conditions))
            category_results = category_query.group_by(ModelDB.category).all()
            result['by_category'] = {row.category or 'uncategorized': row.count for row in category_results}
            
            return result
            
        except SQLAlchemyError as e:
            logger.error("Failed to get model distribution: %s", e)
            return {
                'by_type': {}, 'by_framework': {},
                'by_status': {}, 'by_category': {}
            }
        finally:
            session.close()
    
    def get_top_models(
        self,
        user_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        limit: int = 5,
        order_by: str = 'accuracy'
    ) -> List[Dict[str, Any]]:
        """获取表现最好的模型
        
        Args:
            user_id: 用户ID
            tenant_id: 租户ID
            limit: 返回数量
            order_by: 排序字段 (accuracy/f1_score/created_at)
            
        Returns:
            List[Dict]: 模型列表
        """
        session = self._get_session()
        try:
            from backend.schemas.model_models import ModelDB
            
            conditions = []
            if user_id:
                conditions.append(ModelDB.user_id == user_id)
            if tenant_id:
                conditions.append(ModelDB.tenant_id == tenant_id)
            
            query = session.query(ModelDB)
            if conditions:
                query = query.filter(and_(*conditions))
            
            # 排序
            order_map = {
                'accuracy': desc(ModelDB.accuracy),
                'f1_score': desc(ModelDB.f1_score),
                'created_at': desc(ModelDB.created_at)
            }
            order_clause = order_map.get(order_by, desc(ModelDB.accuracy))
            
            models = query.order_by(order_clause).limit(limit).all()
            
            return [
                {
                    'id': str(m.id),
                    'name': m.name,
                    'model_type': m.model_type,
                    'framework': m.framework,
                    'accuracy': float(m.accuracy or 0),
                    'f1_score': float(m.f1_score or 0),
                    'status': m.status,
                    'created_at': m.created_at.isoformat() if m.created_at else None
                }
                for m in models
            ]
            
        except SQLAlchemyError as e:
            logger.error("Failed to get top models: %s", e)
            return []
        finally:
            session.close()
    
    # =========================================================================
    # 数据集统计查询
    # =========================================================================
    
    def get_dataset_overview(
        self,
        user_id: Optional[str] = None,
        tenant_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """获取数据集概览
        
        Args:
            user_id: 用户ID
            tenant_id: 租户ID
            
        Returns:
            Dict: 数据集概览数据
        """
        session = self._get_session()
        try:
            from backend.schemas.dataset import DatasetEntity
            
            conditions = []
            if user_id:
                conditions.append(DatasetEntity.user_id == user_id)
            if tenant_id:
                conditions.append(DatasetEntity.tenant_id == tenant_id)
            
            base_query = session.query(DatasetEntity)
            if conditions:
                base_query = base_query.filter(and_(*conditions))
            
            total_count = base_query.count()
            
            # 按状态统计
            # pylint: disable=not-callable
            status_query = session.query(
                DatasetEntity.status,
                func.count(DatasetEntity.id).label('count')
            )
            if conditions:
                status_query = status_query.filter(and_(*conditions))
            
            status_results = status_query.group_by(DatasetEntity.status).all()
            status_counts = {row.status: row.count for row in status_results}
            
            return {
                'total_count': total_count,
                'active_count': status_counts.get('active', 0),
                'archived_count': status_counts.get('archived', 0),
                'processing_count': status_counts.get('processing', 0)
            }
            
        except SQLAlchemyError as e:
            logger.error("Failed to get dataset overview: %s", e)
            return {
                'total_count': 0, 'active_count': 0,
                'archived_count': 0, 'processing_count': 0
            }
        finally:
            session.close()
    
    # =========================================================================
    # 用户活动统计
    # =========================================================================
    
    def get_user_activity_summary(
        self,
        user_id: str,
        # pylint: disable=unused-argument
        tenant_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """获取用户活动概要
        
        Args:
            user_id: 用户ID
            tenant_id: 租户ID
            
        Returns:
            Dict: 用户活动概要
        """
        session = self._get_session()
        try:
            from backend.schemas.training_models import TrainingSession
            from backend.schemas.model_models import ModelDB
            from backend.schemas.dataset import DatasetEntity
            
            # 训练统计
            training_count = session.query(TrainingSession).filter(
                TrainingSession.user_id == user_id
            ).count()
            
            # 最后活跃时间
            last_session = session.query(TrainingSession).filter(
                TrainingSession.user_id == user_id
            ).order_by(desc(TrainingSession.created_at)).first()
            
            last_active = last_session.created_at if last_session else None
            
            # 模型数量
            model_count = session.query(ModelDB).filter(
                ModelDB.user_id == user_id
            ).count()
            
            # 数据集数量
            dataset_count = session.query(DatasetEntity).filter(
                DatasetEntity.user_id == user_id
            ).count()
            
            # 最常用训练类型
            # pylint: disable=not-callable
            most_used_type = session.query(
                TrainingSession.training_type,
                func.count(TrainingSession.session_id).label('count')
            ).filter(
                TrainingSession.user_id == user_id
            ).group_by(
                TrainingSession.training_type
            ).order_by(
                desc('count')
            ).first()
            
            return {
                'total_training_count': training_count,
                'total_models_created': model_count,
                'total_datasets_used': dataset_count,
                'last_active_at': last_active.isoformat() if last_active else None,
                'most_used_model_type': most_used_type.training_type if most_used_type else '',
                'avg_training_time_hours': 0.0  # 可以计算
            }
            
        except SQLAlchemyError as e:
            logger.error("Failed to get user activity summary: %s", e)
            return {
                'total_training_count': 0, 'total_models_created': 0,
                'total_datasets_used': 0, 'last_active_at': None,
                'most_used_model_type': '', 'avg_training_time_hours': 0.0
            }
        finally:
            session.close()
    
    # =========================================================================
    # 告警统计
    # =========================================================================
    
    def get_active_alerts_count(
        self,
        tenant_id: Optional[str] = None
    ) -> int:
        """获取活跃告警数量
        
        Args:
            tenant_id: 租户ID
            
        Returns:
            int: 告警数量
        """
        session = self._get_session()
        try:
            # 尝试从性能监控表获取告警
            from backend.schemas.performance_monitoring_models import PerformanceAlertDB as AlertDB
            
            query = session.query(AlertDB).filter(
                AlertDB.status == 'active'
            )
            
            if tenant_id:
                query = query.filter(AlertDB.tenant_id == tenant_id)
            
            return query.count()
            
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.warning("Failed to get alerts count: %s", e)
            return 0
        finally:
            session.close()


    # =========================================================================
    # 训练统计扩展
    # =========================================================================
    
    def get_training_progress_trend(
        self,
        user_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        days: int = 7
    ) -> List[Dict[str, Any]]:
        """获取训练进度趋势数据
        
        按日期统计已完成、运行中、失败的任务数量。
        
        Args:
            user_id: 用户ID
            tenant_id: 租户ID
            days: 统计天数
            
        Returns:
            List[Dict]: 每日趋势数据列表
        """
        session = self._get_session()
        try:
            from backend.schemas.training_models import TrainingSession
            
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=days)
            
            # 构建查询条件
            conditions = [
                TrainingSession.created_at >= start_date,
                TrainingSession.created_at <= end_date
            ]
            if user_id:
                conditions.append(TrainingSession.user_id == user_id)
            if tenant_id:
                conditions.append(TrainingSession.tenant_id == tenant_id)
            
            # 按日期和状态分组查询
            # pylint: disable=not-callable
            date_expr = func.date(TrainingSession.created_at)
            
            query = session.query(
                date_expr.label('date'),
                TrainingSession.status,
                func.count(TrainingSession.session_id).label('count')
            ).filter(
                and_(*conditions)
            ).group_by(
                date_expr,
                TrainingSession.status
            ).order_by(date_expr)
            
            results = query.all()
            
            # 整理数据
            date_data = {}
            for row in results:
                date_str = str(row.date)
                if date_str not in date_data:
                    date_data[date_str] = {
                        'date': date_str,
                        'completed': 0,
                        'running': 0,
                        'failed': 0,
                        'pending': 0,
                        'total': 0
                    }
                
                status = row.status or 'unknown'
                count = row.count or 0
                
                if status in ['completed', 'stopped']:
                    date_data[date_str]['completed'] += count
                elif status in ['running', 'training']:
                    date_data[date_str]['running'] += count
                elif status == 'failed':
                    date_data[date_str]['failed'] += count
                elif status == 'pending':
                    date_data[date_str]['pending'] += count
                
                date_data[date_str]['total'] += count
            
            # 填充缺失日期
            result = []
            current_date = start_date
            while current_date <= end_date:
                date_str = current_date.strftime('%Y-%m-%d')
                if date_str in date_data:
                    result.append(date_data[date_str])
                else:
                    result.append({
                        'date': date_str,
                        'completed': 0,
                        'running': 0,
                        'failed': 0,
                        'pending': 0,
                        'total': 0
                    })
                current_date += timedelta(days=1)
            
            return result
            
        except SQLAlchemyError as e:
            logger.error("Failed to get training progress trend: %s", e)
            return []
        finally:
            session.close()
    
    def get_active_training_sessions(
        self,
        user_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """获取活跃训练会话
        
        Args:
            user_id: 用户ID
            tenant_id: 租户ID
            limit: 返回数量限制
            
        Returns:
            List[Dict]: 活跃训练会话列表
        """
        session = self._get_session()
        try:
            from backend.schemas.training_models import TrainingSession
            
            conditions = [
                TrainingSession.status.in_(['running', 'pending', 'training', 'paused'])
            ]
            if user_id:
                conditions.append(TrainingSession.user_id == user_id)
            if tenant_id:
                conditions.append(TrainingSession.tenant_id == tenant_id)
            
            query = session.query(TrainingSession).filter(
                and_(*conditions)
            ).order_by(
                desc(TrainingSession.created_at)
            ).limit(limit)
            
            sessions = query.all()
            
            result = []
            for s in sessions:
                # 计算进度和预计剩余时间
                progress = getattr(s, 'progress', 0) or 0
                started_at = getattr(s, 'started_at', None) or s.created_at
                
                # 估算剩余时间
                remaining_time = "未知"
                estimated_end = None
                if progress > 0 and started_at:
                    elapsed = (datetime.utcnow() - started_at).total_seconds()
                    if progress < 100:
                        total_estimated = elapsed * 100 / progress
                        remaining_seconds = total_estimated - elapsed
                        if remaining_seconds > 0:
                            hours = int(remaining_seconds // 3600)
                            minutes = int((remaining_seconds % 3600) // 60)
                            remaining_time = f"{hours}小时{minutes}分钟"
                            estimated_end = (datetime.utcnow() + timedelta(seconds=remaining_seconds)).isoformat()
                
                result.append({
                    'id': str(s.session_id),
                    'name': getattr(s, 'name', None) or f'Training {str(s.session_id)[:8]}',
                    'model_type': getattr(s, 'training_type', None) or getattr(s, 'scenario', 'unknown'),
                    'progress': progress,
                    'accuracy': float(getattr(s, 'current_accuracy', 0) or 0),
                    'remaining_time': remaining_time,
                    'status': s.status,
                    'start_time': started_at.isoformat() if started_at else None,
                    'estimated_end': estimated_end,
                    'current_epoch': getattr(s, 'current_epoch', 0),
                    'total_epochs': getattr(s, 'total_epochs', 0)
                })
            
            return result
            
        except SQLAlchemyError as e:
            logger.error("Failed to get active training sessions: %s", e)
            return []
        finally:
            session.close()
    
    def get_training_duration_stats(
        self,
        user_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        days: int = 30
    ) -> Dict[str, Any]:
        """获取训练时长统计
        
        Args:
            user_id: 用户ID
            tenant_id: 租户ID
            days: 统计天数
            
        Returns:
            Dict: 训练时长统计数据
        """
        session = self._get_session()
        try:
            from backend.schemas.training_models import TrainingSession
            
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=days)
            
            conditions = [
                TrainingSession.created_at >= start_date,
                TrainingSession.started_at.isnot(None),
                TrainingSession.completed_at.isnot(None),
                TrainingSession.status.in_(['completed', 'stopped'])
            ]
            if user_id:
                conditions.append(TrainingSession.user_id == user_id)
            if tenant_id:
                conditions.append(TrainingSession.tenant_id == tenant_id)
            
            # 查询所有已完成的训练，计算时长
            # pylint: disable=not-callable
            duration_expr = func.extract('epoch', TrainingSession.completed_at) - \
                           func.extract('epoch', TrainingSession.started_at)
            
            query = session.query(
                func.avg(duration_expr).label('avg_duration'),
                func.min(duration_expr).label('min_duration'),
                func.max(duration_expr).label('max_duration'),
                func.count(TrainingSession.session_id).label('count')
            ).filter(and_(*conditions))
            
            stats = query.first()
            
            # 计算时长分布
            distribution_query = session.query(
                TrainingSession.session_id,
                duration_expr.label('duration_seconds')
            ).filter(and_(*conditions))
            
            durations = [row.duration_seconds for row in distribution_query.all() if row.duration_seconds]
            
            # 分布统计（小时）
            distribution = [
                {'range': '0-1小时', 'count': 0},
                {'range': '1-3小时', 'count': 0},
                {'range': '3-6小时', 'count': 0},
                {'range': '6-12小时', 'count': 0},
                {'range': '12+小时', 'count': 0}
            ]
            
            for d in durations:
                hours = d / 3600
                if hours < 1:
                    distribution[0]['count'] += 1
                elif hours < 3:
                    distribution[1]['count'] += 1
                elif hours < 6:
                    distribution[2]['count'] += 1
                elif hours < 12:
                    distribution[3]['count'] += 1
                else:
                    distribution[4]['count'] += 1
            
            return {
                'avg_duration': float(stats.avg_duration or 0) / 3600,  # 转换为小时
                'min_duration': float(stats.min_duration or 0) / 3600,
                'max_duration': float(stats.max_duration or 0) / 3600,
                'total_count': stats.count or 0,
                'duration_distribution': distribution
            }
            
        except SQLAlchemyError as e:
            logger.error("Failed to get training duration stats: %s", e)
            return {
                'avg_duration': 0,
                'min_duration': 0,
                'max_duration': 0,
                'total_count': 0,
                'duration_distribution': []
            }
        finally:
            session.close()
    
    def get_model_performance_distribution(
        self,
        user_id: Optional[str] = None,
        tenant_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """获取模型性能分布
        
        Args:
            user_id: 用户ID
            tenant_id: 租户ID
            
        Returns:
            Dict: 模型性能分布数据
        """
        session = self._get_session()
        try:
            from backend.schemas.model_models import ModelDB
            
            conditions = []
            if user_id:
                conditions.append(ModelDB.user_id == user_id)
            if tenant_id:
                conditions.append(ModelDB.tenant_id == tenant_id)
            
            # 准确率分布
            accuracy_distribution = [
                {'range': '90-100%', 'count': 0},
                {'range': '80-90%', 'count': 0},
                {'range': '70-80%', 'count': 0},
                {'range': '60-70%', 'count': 0},
                {'range': '<60%', 'count': 0}
            ]
            
            query = session.query(ModelDB.accuracy)
            if conditions:
                query = query.filter(and_(*conditions))
            
            accuracies = [row.accuracy for row in query.all() if row.accuracy is not None]
            
            for acc in accuracies:
                acc_pct = acc * 100 if acc <= 1 else acc
                if acc_pct >= 90:
                    accuracy_distribution[0]['count'] += 1
                elif acc_pct >= 80:
                    accuracy_distribution[1]['count'] += 1
                elif acc_pct >= 70:
                    accuracy_distribution[2]['count'] += 1
                elif acc_pct >= 60:
                    accuracy_distribution[3]['count'] += 1
                else:
                    accuracy_distribution[4]['count'] += 1
            
            # 按模型类型统计
            # pylint: disable=not-callable
            type_query = session.query(
                ModelDB.model_type,
                func.count(ModelDB.id).label('count'),
                func.avg(ModelDB.accuracy).label('avg_accuracy')
            )
            if conditions:
                type_query = type_query.filter(and_(*conditions))
            
            type_results = type_query.group_by(ModelDB.model_type).all()
            
            model_types = [
                {
                    'type': row.model_type or 'unknown',
                    'count': row.count,
                    'avg_accuracy': round(float(row.avg_accuracy or 0) * 100, 2) if row.avg_accuracy and row.avg_accuracy <= 1 else round(float(row.avg_accuracy or 0), 2)
                }
                for row in type_results
            ]
            
            return {
                'accuracy_distribution': accuracy_distribution,
                'model_types': model_types,
                'total_models': len(accuracies)
            }
            
        except SQLAlchemyError as e:
            logger.error("Failed to get model performance distribution: %s", e)
            return {
                'accuracy_distribution': [],
                'model_types': [],
                'total_models': 0
            }
        finally:
            session.close()


# 全局仓库实例
_dashboard_repository: Optional[DashboardRepository] = None


def get_dashboard_repository() -> DashboardRepository:
    """获取仪表盘仓库单例
    
    Returns:
        DashboardRepository: 仪表盘仓库实例
    """
    global _dashboard_repository
    if _dashboard_repository is None:
        _dashboard_repository = DashboardRepository()
    return _dashboard_repository

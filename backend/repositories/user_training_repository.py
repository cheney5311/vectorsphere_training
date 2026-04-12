"""用户训练仓库

提供用户训练相关的数据访问接口，包括训练会话查询、统计聚合等功能。

仓库层负责：
1. 封装数据库查询逻辑
2. 实现数据聚合和统计
3. 提供统一的数据访问接口
4. 处理数据库事务
"""

import logging
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime, timedelta
from sqlalchemy import func, and_, or_, desc, asc, case, distinct
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError

logger = logging.getLogger(__name__)


class UserTrainingRepository:
    """用户训练仓库
    
    提供用户训练相关的数据访问方法。
    
    主要功能:
        - 获取用户训练概览数据
        - 查询用户训练会话
        - 统计训练数据
        - 获取训练趋势
    """
    
    def __init__(self, db_manager=None):
        """初始化用户训练仓库
        
        Args:
            db_manager: 数据库管理器实例，如果为None则自动获取
        """
        if db_manager is None:
            from backend.modules.database.manager import get_database_manager
            db_manager = get_database_manager()
        self._db_manager = db_manager
    
    def _get_session(self) -> Session:
        """获取数据库会话"""
        return self._db_manager.get_session()
    
    # =========================================================================
    # 概览相关
    # =========================================================================
    
    def get_user_training_overview(
        self,
        user_id: str,
        tenant_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """获取用户训练概览数据
        
        汇总用户的训练活动数据。
        
        Args:
            user_id: 用户ID
            tenant_id: 租户ID（可选）
            
        Returns:
            Dict: 概览数据，包含:
                - active_count: 活跃会话数
                - completed_count: 完成会话数
                - total_models: 模型总数
                - avg_accuracy: 平均准确率
                - success_rate: 成功率
                - total_training_hours: 总训练时长
        """
        session = self._get_session()
        try:
            from backend.schemas.training_models import TrainingSession, TrainingProgress
            
            # 构建基础条件
            conditions = [TrainingSession.user_id == user_id]
            if tenant_id:
                conditions.append(TrainingSession.tenant_id == tenant_id)
            
            base_query = session.query(TrainingSession).filter(and_(*conditions))
            
            # 活跃会话数
            active_count = base_query.filter(
                TrainingSession.status.in_(['running', 'pending', 'training', 'paused'])
            ).count()
            
            # 完成会话数
            completed_count = base_query.filter(
                TrainingSession.status == 'completed'
            ).count()
            
            # 失败会话数
            failed_count = base_query.filter(
                TrainingSession.status == 'failed'
            ).count()
            
            # 总会话数
            total_count = base_query.count()
            
            # 模型总数（按 model_id 去重）
            total_models = session.query(
                func.count(distinct(TrainingSession.model_id))
            ).filter(
                and_(*conditions),
                TrainingSession.model_id.isnot(None)
            ).scalar() or 0
            
            # 平均准确率（从进度记录中获取）
            accuracy_query = session.query(
                func.avg(TrainingProgress.accuracy)
            ).join(
                TrainingSession,
                TrainingProgress.session_id == TrainingSession.session_id
            ).filter(
                and_(*conditions),
                TrainingProgress.accuracy.isnot(None),
                TrainingSession.status == 'completed'
            )
            avg_accuracy = accuracy_query.scalar() or 0.0
            
            # 计算成功率
            success_rate = 0.0
            if total_count > 0:
                success_rate = (completed_count / total_count) * 100
            
            # 计算总训练时长
            duration_query = session.query(
                func.sum(
                    func.extract('epoch', TrainingSession.completed_at) -
                    func.extract('epoch', TrainingSession.started_at)
                )
            ).filter(
                and_(*conditions),
                TrainingSession.started_at.isnot(None),
                TrainingSession.completed_at.isnot(None)
            )
            total_seconds = duration_query.scalar() or 0
            total_training_hours = total_seconds / 3600
            
            return {
                'active_count': active_count,
                'completed_count': completed_count,
                'failed_count': failed_count,
                'total_count': total_count,
                'total_models': total_models,
                'avg_accuracy': float(avg_accuracy) * 100 if avg_accuracy <= 1 else float(avg_accuracy),
                'success_rate': round(success_rate, 2),
                'total_training_hours': round(total_training_hours, 2)
            }
            
        except SQLAlchemyError as e:
            logger.error(f"Failed to get user training overview: {e}")
            return {
                'active_count': 0,
                'completed_count': 0,
                'failed_count': 0,
                'total_count': 0,
                'total_models': 0,
                'avg_accuracy': 0.0,
                'success_rate': 0.0,
                'total_training_hours': 0.0
            }
        finally:
            session.close()
    
    # =========================================================================
    # 会话相关
    # =========================================================================
    
    def get_recent_sessions(
        self,
        user_id: str,
        limit: int = 5,
        tenant_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """获取用户最近的训练会话
        
        Args:
            user_id: 用户ID
            limit: 返回数量
            tenant_id: 租户ID（可选）
            
        Returns:
            List[Dict]: 会话列表
        """
        session = self._get_session()
        try:
            from backend.schemas.training_models import TrainingSession, TrainingProgress
            
            conditions = [TrainingSession.user_id == user_id]
            if tenant_id:
                conditions.append(TrainingSession.tenant_id == tenant_id)
            
            # 查询最近的会话
            sessions = session.query(TrainingSession).filter(
                and_(*conditions)
            ).order_by(
                desc(TrainingSession.created_at)
            ).limit(limit).all()
            
            result = []
            for s in sessions:
                # 获取最新进度
                latest_progress = session.query(TrainingProgress).filter(
                    TrainingProgress.session_id == s.session_id
                ).order_by(desc(TrainingProgress.created_at)).first()
                
                session_data = {
                    'session_id': s.session_id,
                    'id': s.session_id,
                    'name': None,
                    'training_type': s.training_type,
                    'status': s.status,
                    'progress': s.progress or 0,
                    'accuracy': 0.0,
                    'loss': 0.0,
                    'config': s.config,
                    'created_at': s.created_at,
                    'started_at': s.started_at,
                    'completed_at': s.completed_at,
                    'error_message': s.error_message
                }
                
                # 从配置中获取名称
                if s.config and isinstance(s.config, dict):
                    session_data['name'] = s.config.get('session_name')
                
                # 从进度中获取准确率和损失
                if latest_progress:
                    session_data['accuracy'] = latest_progress.accuracy or 0
                    session_data['loss'] = latest_progress.loss or 0
                    session_data['current_epoch'] = latest_progress.epoch
                    if latest_progress.progress is not None:
                        session_data['progress'] = latest_progress.progress
                
                result.append(session_data)
            
            return result
            
        except SQLAlchemyError as e:
            logger.error(f"Failed to get recent sessions: {e}")
            return []
        finally:
            session.close()
    
    def get_user_sessions(
        self,
        user_id: str,
        offset: int = 0,
        limit: int = 10,
        status: Optional[str] = None,
        model_type: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        sort_by: str = 'created_at',
        sort_order: str = 'desc',
        tenant_id: Optional[str] = None
    ) -> Tuple[List[Dict[str, Any]], int]:
        """获取用户训练会话列表（支持分页和筛选）
        
        Args:
            user_id: 用户ID
            offset: 偏移量
            limit: 返回数量
            status: 状态筛选
            model_type: 模型类型筛选
            start_date: 开始日期
            end_date: 结束日期
            sort_by: 排序字段
            sort_order: 排序方向
            tenant_id: 租户ID
            
        Returns:
            Tuple[List[Dict], int]: (会话列表, 总数)
        """
        session = self._get_session()
        try:
            from backend.schemas.training_models import TrainingSession, TrainingProgress
            
            # 构建条件
            conditions = [TrainingSession.user_id == user_id]
            if tenant_id:
                conditions.append(TrainingSession.tenant_id == tenant_id)
            if status:
                conditions.append(TrainingSession.status == status)
            if model_type:
                conditions.append(TrainingSession.training_type == model_type)
            if start_date:
                conditions.append(TrainingSession.created_at >= start_date)
            if end_date:
                conditions.append(TrainingSession.created_at <= end_date)
            
            base_query = session.query(TrainingSession).filter(and_(*conditions))
            
            # 获取总数
            total = base_query.count()
            
            # 排序
            sort_column = getattr(TrainingSession, sort_by, TrainingSession.created_at)
            if sort_order == 'asc':
                base_query = base_query.order_by(asc(sort_column))
            else:
                base_query = base_query.order_by(desc(sort_column))
            
            # 分页
            sessions = base_query.offset(offset).limit(limit).all()
            
            result = []
            for s in sessions:
                # 获取最新进度
                latest_progress = session.query(TrainingProgress).filter(
                    TrainingProgress.session_id == s.session_id
                ).order_by(desc(TrainingProgress.created_at)).first()
                
                session_data = {
                    'session_id': s.session_id,
                    'id': s.session_id,
                    'name': None,
                    'training_type': s.training_type,
                    'model_type': s.training_type,
                    'status': s.status,
                    'progress': s.progress or 0,
                    'accuracy': 0.0,
                    'loss': 0.0,
                    'config': s.config,
                    'result': s.result,
                    'created_at': s.created_at,
                    'started_at': s.started_at,
                    'completed_at': s.completed_at,
                    'error_message': s.error_message
                }
                
                if s.config and isinstance(s.config, dict):
                    session_data['name'] = s.config.get('session_name')
                
                if latest_progress:
                    session_data['accuracy'] = latest_progress.accuracy or 0
                    session_data['loss'] = latest_progress.loss or 0
                    session_data['current_epoch'] = latest_progress.epoch
                    session_data['total_epochs'] = latest_progress.metrics.get('total_epochs', 0) if latest_progress.metrics else 0
                
                result.append(session_data)
            
            return result, total
            
        except SQLAlchemyError as e:
            logger.error(f"Failed to get user sessions: {e}")
            return [], 0
        finally:
            session.close()
    
    def get_session_detail(
        self,
        session_id: str,
        user_id: str,
        tenant_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """获取训练会话详情
        
        Args:
            session_id: 会话ID
            user_id: 用户ID
            tenant_id: 租户ID
            
        Returns:
            Dict: 会话详情
        """
        session = self._get_session()
        try:
            from backend.schemas.training_models import TrainingSession, TrainingProgress
            
            conditions = [
                TrainingSession.session_id == session_id,
                TrainingSession.user_id == user_id
            ]
            if tenant_id:
                conditions.append(TrainingSession.tenant_id == tenant_id)
            
            training_session = session.query(TrainingSession).filter(
                and_(*conditions)
            ).first()
            
            if not training_session:
                return None
            
            # 获取所有进度记录
            progress_records = session.query(TrainingProgress).filter(
                TrainingProgress.session_id == session_id
            ).order_by(asc(TrainingProgress.created_at)).all()
            
            # 构建详情
            detail = {
                'session_id': training_session.session_id,
                'user_id': training_session.user_id,
                'model_id': training_session.model_id,
                'dataset_id': training_session.dataset_id,
                'training_type': training_session.training_type,
                'status': training_session.status,
                'progress': training_session.progress,
                'config': training_session.config,
                'result': training_session.result,
                'error_message': training_session.error_message,
                'created_at': training_session.created_at.isoformat() if training_session.created_at else None,
                'started_at': training_session.started_at.isoformat() if training_session.started_at else None,
                'completed_at': training_session.completed_at.isoformat() if training_session.completed_at else None,
                'progress_history': []
            }
            
            # 添加进度历史
            for p in progress_records:
                detail['progress_history'].append({
                    'epoch': p.epoch,
                    'step': p.step,
                    'loss': p.loss,
                    'accuracy': p.accuracy,
                    'learning_rate': p.learning_rate,
                    'metrics': p.metrics,
                    'timestamp': p.created_at.isoformat() if p.created_at else None
                })
            
            return detail
            
        except SQLAlchemyError as e:
            logger.error(f"Failed to get session detail: {e}")
            return None
        finally:
            session.close()
    
    def get_active_sessions(
        self,
        user_id: str,
        tenant_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """获取用户活跃训练会话
        
        Args:
            user_id: 用户ID
            tenant_id: 租户ID
            
        Returns:
            List[Dict]: 活跃会话列表
        """
        session = self._get_session()
        try:
            from backend.schemas.training_models import TrainingSession, TrainingProgress
            
            conditions = [
                TrainingSession.user_id == user_id,
                TrainingSession.status.in_(['running', 'pending', 'training', 'paused'])
            ]
            if tenant_id:
                conditions.append(TrainingSession.tenant_id == tenant_id)
            
            sessions = session.query(TrainingSession).filter(
                and_(*conditions)
            ).order_by(desc(TrainingSession.created_at)).all()
            
            result = []
            for s in sessions:
                latest_progress = session.query(TrainingProgress).filter(
                    TrainingProgress.session_id == s.session_id
                ).order_by(desc(TrainingProgress.created_at)).first()
                
                session_data = {
                    'session_id': s.session_id,
                    'name': s.config.get('session_name') if s.config else None,
                    'training_type': s.training_type,
                    'status': s.status,
                    'progress': s.progress or 0,
                    'created_at': s.created_at,
                    'started_at': s.started_at
                }
                
                if latest_progress:
                    session_data['accuracy'] = latest_progress.accuracy or 0
                    session_data['loss'] = latest_progress.loss or 0
                    session_data['current_epoch'] = latest_progress.epoch
                
                result.append(session_data)
            
            return result
            
        except SQLAlchemyError as e:
            logger.error(f"Failed to get active sessions: {e}")
            return []
        finally:
            session.close()
    
    # =========================================================================
    # 统计相关
    # =========================================================================
    
    def get_user_statistics(
        self,
        user_id: str,
        days: int = 30,
        tenant_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """获取用户训练统计
        
        Args:
            user_id: 用户ID
            days: 统计天数
            tenant_id: 租户ID
            
        Returns:
            Dict: 统计数据
        """
        session = self._get_session()
        try:
            from backend.schemas.training_models import TrainingSession, TrainingProgress
            
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=days)
            
            conditions = [
                TrainingSession.user_id == user_id,
                TrainingSession.created_at >= start_date
            ]
            if tenant_id:
                conditions.append(TrainingSession.tenant_id == tenant_id)
            
            base_query = session.query(TrainingSession).filter(and_(*conditions))
            
            # 各状态数量
            total_count = base_query.count()
            completed_count = base_query.filter(TrainingSession.status == 'completed').count()
            running_count = base_query.filter(TrainingSession.status.in_(['running', 'training'])).count()
            pending_count = base_query.filter(TrainingSession.status == 'pending').count()
            failed_count = base_query.filter(TrainingSession.status == 'failed').count()
            cancelled_count = base_query.filter(TrainingSession.status == 'cancelled').count()
            
            # 成功率
            success_rate = (completed_count / total_count * 100) if total_count > 0 else 0
            
            # 平均训练时间（分钟）
            duration_query = session.query(
                func.avg(
                    func.extract('epoch', TrainingSession.completed_at) -
                    func.extract('epoch', TrainingSession.started_at)
                )
            ).filter(
                and_(*conditions),
                TrainingSession.started_at.isnot(None),
                TrainingSession.completed_at.isnot(None),
                TrainingSession.status == 'completed'
            )
            avg_duration_seconds = duration_query.scalar() or 0
            avg_training_time = avg_duration_seconds / 60
            
            # 总训练时间（小时）
            total_duration_query = session.query(
                func.sum(
                    func.extract('epoch', TrainingSession.completed_at) -
                    func.extract('epoch', TrainingSession.started_at)
                )
            ).filter(
                and_(*conditions),
                TrainingSession.started_at.isnot(None),
                TrainingSession.completed_at.isnot(None)
            )
            total_duration_seconds = total_duration_query.scalar() or 0
            total_training_time = total_duration_seconds / 3600
            
            # 准确率统计
            accuracy_stats = session.query(
                func.avg(TrainingProgress.accuracy),
                func.max(TrainingProgress.accuracy)
            ).join(
                TrainingSession,
                TrainingProgress.session_id == TrainingSession.session_id
            ).filter(
                and_(*conditions),
                TrainingProgress.accuracy.isnot(None)
            ).first()
            
            avg_accuracy = float(accuracy_stats[0] or 0)
            best_accuracy = float(accuracy_stats[1] or 0)
            
            # 损失统计
            loss_stats = session.query(
                func.avg(TrainingProgress.loss),
                func.min(TrainingProgress.loss)
            ).join(
                TrainingSession,
                TrainingProgress.session_id == TrainingSession.session_id
            ).filter(
                and_(*conditions),
                TrainingProgress.loss.isnot(None),
                TrainingProgress.loss > 0
            ).first()
            
            avg_loss = float(loss_stats[0] or 0)
            best_loss = float(loss_stats[1] or 0)
            
            return {
                'total_count': total_count,
                'completed_count': completed_count,
                'running_count': running_count,
                'pending_count': pending_count,
                'failed_count': failed_count,
                'cancelled_count': cancelled_count,
                'success_rate': round(success_rate, 2),
                'avg_training_time': round(avg_training_time, 2),
                'total_training_time': round(total_training_time, 2),
                'avg_accuracy': avg_accuracy * 100 if avg_accuracy <= 1 else avg_accuracy,
                'best_accuracy': best_accuracy * 100 if best_accuracy <= 1 else best_accuracy,
                'avg_loss': avg_loss,
                'best_loss': best_loss
            }
            
        except SQLAlchemyError as e:
            logger.error(f"Failed to get user statistics: {e}")
            return {
                'total_count': 0,
                'completed_count': 0,
                'running_count': 0,
                'pending_count': 0,
                'failed_count': 0,
                'cancelled_count': 0,
                'success_rate': 0.0,
                'avg_training_time': 0.0,
                'total_training_time': 0.0,
                'avg_accuracy': 0.0,
                'best_accuracy': 0.0,
                'avg_loss': 0.0,
                'best_loss': 0.0
            }
        finally:
            session.close()
    
    def get_training_trend(
        self,
        user_id: str,
        days: int = 7,
        tenant_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """获取训练趋势数据
        
        Args:
            user_id: 用户ID
            days: 统计天数
            tenant_id: 租户ID
            
        Returns:
            List[Dict]: 每日趋势数据
        """
        session = self._get_session()
        try:
            from backend.schemas.training_models import TrainingSession
            
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=days)
            
            conditions = [
                TrainingSession.user_id == user_id,
                TrainingSession.created_at >= start_date,
                TrainingSession.created_at <= end_date
            ]
            if tenant_id:
                conditions.append(TrainingSession.tenant_id == tenant_id)
            
            # 按日期和状态分组
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
            logger.error(f"Failed to get training trend: {e}")
            return []
        finally:
            session.close()
    
    def get_model_performance_ranking(
        self,
        user_id: str,
        limit: int = 10,
        tenant_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """获取模型性能排行
        
        Args:
            user_id: 用户ID
            limit: 返回数量
            tenant_id: 租户ID
            
        Returns:
            List[Dict]: 模型性能列表
        """
        session = self._get_session()
        try:
            from backend.schemas.training_models import TrainingSession, TrainingProgress
            
            conditions = [
                TrainingSession.user_id == user_id,
                TrainingSession.model_id.isnot(None)
            ]
            if tenant_id:
                conditions.append(TrainingSession.tenant_id == tenant_id)
            
            # 按模型分组，获取最佳准确率
            subquery = session.query(
                TrainingSession.model_id,
                func.max(TrainingProgress.accuracy).label('best_accuracy'),
                func.min(TrainingProgress.loss).label('best_loss'),
                func.count(distinct(TrainingSession.session_id)).label('training_count'),
                func.max(TrainingSession.created_at).label('last_trained')
            ).join(
                TrainingProgress,
                TrainingProgress.session_id == TrainingSession.session_id
            ).filter(
                and_(*conditions),
                TrainingProgress.accuracy.isnot(None)
            ).group_by(
                TrainingSession.model_id
            ).order_by(
                desc('best_accuracy')
            ).limit(limit).subquery()
            
            # 获取结果
            results = session.query(
                subquery.c.model_id,
                subquery.c.best_accuracy,
                subquery.c.best_loss,
                subquery.c.training_count,
                subquery.c.last_trained
            ).all()
            
            return [
                {
                    'model_id': r.model_id,
                    'model_name': r.model_id,  # 如果有模型表可以关联获取名称
                    'model_type': 'unknown',
                    'best_accuracy': float(r.best_accuracy or 0),
                    'best_loss': float(r.best_loss or 0),
                    'training_count': r.training_count or 0,
                    'last_trained': r.last_trained.isoformat() if r.last_trained else None
                }
                for r in results
            ]
            
        except SQLAlchemyError as e:
            logger.error(f"Failed to get model performance ranking: {e}")
            return []
        finally:
            session.close()
    
    def get_training_duration_stats(
        self,
        user_id: str,
        days: int = 30,
        tenant_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """获取训练时长统计
        
        Args:
            user_id: 用户ID
            days: 统计天数
            tenant_id: 租户ID
            
        Returns:
            Dict: 时长统计数据
        """
        session = self._get_session()
        try:
            from backend.schemas.training_models import TrainingSession
            
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=days)
            
            conditions = [
                TrainingSession.user_id == user_id,
                TrainingSession.created_at >= start_date,
                TrainingSession.started_at.isnot(None),
                TrainingSession.completed_at.isnot(None),
                TrainingSession.status.in_(['completed', 'stopped'])
            ]
            if tenant_id:
                conditions.append(TrainingSession.tenant_id == tenant_id)
            
            # 时长统计
            duration_expr = func.extract('epoch', TrainingSession.completed_at) - \
                           func.extract('epoch', TrainingSession.started_at)
            
            stats = session.query(
                func.avg(duration_expr).label('avg_duration'),
                func.min(duration_expr).label('min_duration'),
                func.max(duration_expr).label('max_duration'),
                func.sum(duration_expr).label('total_duration'),
                func.count(TrainingSession.session_id).label('count')
            ).filter(and_(*conditions)).first()
            
            # 计算时长分布
            durations = session.query(
                duration_expr.label('duration_seconds')
            ).filter(and_(*conditions)).all()
            
            distribution = [
                {'range': '0-30分钟', 'count': 0},
                {'range': '30-60分钟', 'count': 0},
                {'range': '1-2小时', 'count': 0},
                {'range': '2-4小时', 'count': 0},
                {'range': '4+小时', 'count': 0}
            ]
            
            for d in durations:
                if d.duration_seconds:
                    minutes = d.duration_seconds / 60
                    if minutes < 30:
                        distribution[0]['count'] += 1
                    elif minutes < 60:
                        distribution[1]['count'] += 1
                    elif minutes < 120:
                        distribution[2]['count'] += 1
                    elif minutes < 240:
                        distribution[3]['count'] += 1
                    else:
                        distribution[4]['count'] += 1
            
            return {
                'avgDuration': round(float(stats.avg_duration or 0) / 60, 2),  # 分钟
                'minDuration': round(float(stats.min_duration or 0) / 60, 2),
                'maxDuration': round(float(stats.max_duration or 0) / 60, 2),
                'totalDuration': round(float(stats.total_duration or 0) / 3600, 2),  # 小时
                'totalCount': stats.count or 0,
                'distribution': distribution
            }
            
        except SQLAlchemyError as e:
            logger.error(f"Failed to get training duration stats: {e}")
            return {
                'avgDuration': 0,
                'minDuration': 0,
                'maxDuration': 0,
                'totalDuration': 0,
                'totalCount': 0,
                'distribution': []
            }
        finally:
            session.close()


# =============================================================================
# 全局仓库实例
# =============================================================================

_user_training_repository: Optional[UserTrainingRepository] = None


def get_user_training_repository() -> UserTrainingRepository:
    """获取用户训练仓库单例
    
    Returns:
        UserTrainingRepository: 用户训练仓库实例
    """
    global _user_training_repository
    if _user_training_repository is None:
        _user_training_repository = UserTrainingRepository()
    return _user_training_repository


def reset_user_training_repository():
    """重置用户训练仓库（用于测试）"""
    global _user_training_repository
    _user_training_repository = None

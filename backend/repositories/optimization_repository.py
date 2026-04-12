#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""资源优化模块数据访问层

提供优化会话、优化建议、资源指标、性能分析报告、资源告警等实体的数据访问操作。
支持内存存储和数据库持久化两种模式。
"""

import logging
import threading
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict

logger = logging.getLogger(__name__)


# ==================== 优化会话仓库 ====================

class OptimizationSessionRepository:
    """优化会话仓库
    
    管理优化会话的CRUD操作和查询
    """
    
    def __init__(self, use_memory_storage: bool = True):
        self._use_memory = use_memory_storage
        self._lock = threading.RLock()
        
        # 内存存储
        self._sessions: Dict[str, Dict[str, Any]] = {}
        
        if not use_memory_storage:
            self._init_database()
    
    def _init_database(self):
        """初始化数据库连接"""
        try:
            from backend.database.db_pool import DatabasePool
            self._db_pool = DatabasePool()
            logger.info("OptimizationSessionRepository database initialized")
        except Exception as e:
            logger.warning(f"Failed to init database, using memory: {e}")
            self._use_memory = True
    
    def create(self, session: Dict[str, Any]) -> Dict[str, Any]:
        """创建优化会话"""
        if not session.get('id'):
            session['id'] = f"opt_session_{uuid.uuid4().hex[:12]}"
        
        if not session.get('created_at'):
            session['created_at'] = datetime.utcnow().isoformat()
        
        if self._use_memory:
            with self._lock:
                self._sessions[session['id']] = session.copy()
        else:
            self._db_create(session)
        
        logger.info(f"Created optimization session: {session['id']}")
        return session
    
    def _db_create(self, session: Dict[str, Any]):
        """数据库创建"""
        try:
            from backend.schemas.optimization_models import OptimizationSessionModel
            db_session = self._db_pool.get_session()
            
            model = OptimizationSessionModel(
                id=session['id'],
                tenant_id=session.get('tenant_id'),
                name=session.get('name'),
                description=session.get('description'),
                strategy=session.get('strategy', 'balanced'),
                strategy_config=session.get('strategy_config', {}),
                status=session.get('status', 'idle'),
                progress=session.get('progress', 0.0),
                target_resources=session.get('target_resources', []),
                created_by=session.get('created_by')
            )
            
            db_session.add(model)
            db_session.commit()
        except Exception as e:
            logger.error(f"Database create failed: {e}")
            # 回退到内存存储
            with self._lock:
                self._sessions[session['id']] = session.copy()
    
    def get_by_id(self, session_id: str) -> Optional[Dict[str, Any]]:
        """根据ID获取会话"""
        if self._use_memory:
            with self._lock:
                return self._sessions.get(session_id, {}).copy() if session_id in self._sessions else None
        else:
            return self._db_get_by_id(session_id)
    
    def _db_get_by_id(self, session_id: str) -> Optional[Dict[str, Any]]:
        """数据库获取"""
        try:
            from backend.schemas.optimization_models import OptimizationSessionModel
            db_session = self._db_pool.get_session()
            
            model = db_session.query(OptimizationSessionModel).filter_by(id=session_id).first()
            return model.to_dict() if model else None
        except Exception as e:
            logger.error(f"Database get failed: {e}")
            with self._lock:
                return self._sessions.get(session_id, {}).copy() if session_id in self._sessions else None
    
    def update(self, session_id: str, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """更新会话"""
        if self._use_memory:
            with self._lock:
                if session_id not in self._sessions:
                    return None
                self._sessions[session_id].update(updates)
                return self._sessions[session_id].copy()
        else:
            return self._db_update(session_id, updates)
    
    def _db_update(self, session_id: str, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """数据库更新"""
        try:
            from backend.schemas.optimization_models import OptimizationSessionModel
            db_session = self._db_pool.get_session()
            
            model = db_session.query(OptimizationSessionModel).filter_by(id=session_id).first()
            if not model:
                return None
            
            for key, value in updates.items():
                if hasattr(model, key):
                    setattr(model, key, value)
            
            db_session.commit()
            return model.to_dict()
        except Exception as e:
            logger.error(f"Database update failed: {e}")
            with self._lock:
                if session_id in self._sessions:
                    self._sessions[session_id].update(updates)
                    return self._sessions[session_id].copy()
            return None
    
    def delete(self, session_id: str) -> bool:
        """删除会话"""
        if self._use_memory:
            with self._lock:
                if session_id in self._sessions:
                    del self._sessions[session_id]
                    return True
                return False
        else:
            return self._db_delete(session_id)
    
    def _db_delete(self, session_id: str) -> bool:
        """数据库删除"""
        try:
            from backend.schemas.optimization_models import OptimizationSessionModel
            db_session = self._db_pool.get_session()
            
            result = db_session.query(OptimizationSessionModel).filter_by(id=session_id).delete()
            db_session.commit()
            return result > 0
        except Exception as e:
            logger.error(f"Database delete failed: {e}")
            return False
    
    def list_sessions(
        self,
        tenant_id: str = None,
        status: str = None,
        strategy: str = None,
        limit: int = 100,
        offset: int = 0
    ) -> Tuple[List[Dict[str, Any]], int]:
        """列出会话"""
        if self._use_memory:
            with self._lock:
                sessions = list(self._sessions.values())
                
                # 过滤
                if tenant_id:
                    sessions = [s for s in sessions if s.get('tenant_id') == tenant_id]
                if status:
                    sessions = [s for s in sessions if s.get('status') == status]
                if strategy:
                    sessions = [s for s in sessions if s.get('strategy') == strategy]
                
                # 排序
                sessions.sort(key=lambda x: x.get('created_at', ''), reverse=True)
                
                total = len(sessions)
                return sessions[offset:offset + limit], total
        else:
            return self._db_list_sessions(tenant_id, status, strategy, limit, offset)
    
    def _db_list_sessions(
        self,
        tenant_id: str,
        status: str,
        strategy: str,
        limit: int,
        offset: int
    ) -> Tuple[List[Dict[str, Any]], int]:
        """数据库列出"""
        try:
            from backend.schemas.optimization_models import OptimizationSessionModel
            db_session = self._db_pool.get_session()
            
            query = db_session.query(OptimizationSessionModel)
            
            if tenant_id:
                query = query.filter_by(tenant_id=tenant_id)
            if status:
                query = query.filter_by(status=status)
            if strategy:
                query = query.filter_by(strategy=strategy)
            
            total = query.count()
            models = query.order_by(OptimizationSessionModel.created_at.desc()).offset(offset).limit(limit).all()
            
            return [m.to_dict() for m in models], total
        except Exception as e:
            logger.error(f"Database list failed: {e}")
            return [], 0
    
    def get_active_session(self, tenant_id: str = None) -> Optional[Dict[str, Any]]:
        """获取活跃的优化会话"""
        active_statuses = ['analyzing', 'optimizing']
        
        if self._use_memory:
            with self._lock:
                for session in self._sessions.values():
                    if session.get('status') in active_statuses:
                        if not tenant_id or session.get('tenant_id') == tenant_id:
                            return session.copy()
                return None
        else:
            try:
                from backend.schemas.optimization_models import OptimizationSessionModel, OptimizationStatusEnum
                db_session = self._db_pool.get_session()
                
                query = db_session.query(OptimizationSessionModel).filter(
                    OptimizationSessionModel.status.in_([
                        OptimizationStatusEnum.ANALYZING,
                        OptimizationStatusEnum.OPTIMIZING
                    ])
                )
                
                if tenant_id:
                    query = query.filter_by(tenant_id=tenant_id)
                
                model = query.first()
                return model.to_dict() if model else None
            except Exception as e:
                logger.error(f"Get active session failed: {e}")
                return None


# ==================== 优化建议仓库 ====================

class OptimizationRecommendationRepository:
    """优化建议仓库
    
    管理优化建议的CRUD操作和查询
    """
    
    def __init__(self, use_memory_storage: bool = True):
        self._use_memory = use_memory_storage
        self._lock = threading.RLock()
        
        # 内存存储
        self._recommendations: Dict[str, Dict[str, Any]] = {}
        
        if not use_memory_storage:
            self._init_database()
    
    def _init_database(self):
        """初始化数据库连接"""
        try:
            from backend.database.db_pool import DatabasePool
            self._db_pool = DatabasePool()
            logger.info("OptimizationRecommendationRepository database initialized")
        except Exception as e:
            logger.warning(f"Failed to init database, using memory: {e}")
            self._use_memory = True
    
    def create(self, recommendation: Dict[str, Any]) -> Dict[str, Any]:
        """创建优化建议"""
        if not recommendation.get('id'):
            recommendation['id'] = f"opt_rec_{uuid.uuid4().hex[:12]}"
        
        if not recommendation.get('created_at'):
            recommendation['created_at'] = datetime.utcnow().isoformat()
        
        if self._use_memory:
            with self._lock:
                self._recommendations[recommendation['id']] = recommendation.copy()
        else:
            self._db_create(recommendation)
        
        logger.debug(f"Created recommendation: {recommendation['id']}")
        return recommendation
    
    def _db_create(self, recommendation: Dict[str, Any]):
        """数据库创建"""
        try:
            from backend.schemas.optimization_models import OptimizationRecommendationModel
            db_session = self._db_pool.get_session()
            
            model = OptimizationRecommendationModel(
                id=recommendation['id'],
                tenant_id=recommendation.get('tenant_id'),
                session_id=recommendation.get('session_id'),
                title=recommendation.get('title'),
                description=recommendation.get('description'),
                category=recommendation.get('category'),
                priority=recommendation.get('priority', 'medium'),
                confidence=recommendation.get('confidence', 0.5),
                status=recommendation.get('status', 'pending'),
                action=recommendation.get('action'),
                action_params=recommendation.get('action_params', {}),
                estimated_impact=recommendation.get('estimated_impact'),
                estimated_savings_percent=recommendation.get('estimated_savings_percent', 0.0),
                risk_level=recommendation.get('risk_level', 'low'),
                current_value=recommendation.get('current_value'),
                recommended_value=recommendation.get('recommended_value'),
                threshold=recommendation.get('threshold')
            )
            
            db_session.add(model)
            db_session.commit()
        except Exception as e:
            logger.error(f"Database create failed: {e}")
            with self._lock:
                self._recommendations[recommendation['id']] = recommendation.copy()
    
    def get_by_id(self, recommendation_id: str) -> Optional[Dict[str, Any]]:
        """根据ID获取建议"""
        if self._use_memory:
            with self._lock:
                rec = self._recommendations.get(recommendation_id)
                return rec.copy() if rec else None
        else:
            return self._db_get_by_id(recommendation_id)
    
    def _db_get_by_id(self, recommendation_id: str) -> Optional[Dict[str, Any]]:
        """数据库获取"""
        try:
            from backend.schemas.optimization_models import OptimizationRecommendationModel
            db_session = self._db_pool.get_session()
            
            model = db_session.query(OptimizationRecommendationModel).filter_by(id=recommendation_id).first()
            return model.to_dict() if model else None
        except Exception as e:
            logger.error(f"Database get failed: {e}")
            with self._lock:
                rec = self._recommendations.get(recommendation_id)
                return rec.copy() if rec else None
    
    def update(self, recommendation_id: str, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """更新建议"""
        if self._use_memory:
            with self._lock:
                if recommendation_id not in self._recommendations:
                    return None
                self._recommendations[recommendation_id].update(updates)
                return self._recommendations[recommendation_id].copy()
        else:
            return self._db_update(recommendation_id, updates)
    
    def _db_update(self, recommendation_id: str, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """数据库更新"""
        try:
            from backend.schemas.optimization_models import OptimizationRecommendationModel
            db_session = self._db_pool.get_session()
            
            model = db_session.query(OptimizationRecommendationModel).filter_by(id=recommendation_id).first()
            if not model:
                return None
            
            for key, value in updates.items():
                if hasattr(model, key):
                    setattr(model, key, value)
            
            db_session.commit()
            return model.to_dict()
        except Exception as e:
            logger.error(f"Database update failed: {e}")
            return None
    
    def delete(self, recommendation_id: str) -> bool:
        """删除建议"""
        if self._use_memory:
            with self._lock:
                if recommendation_id in self._recommendations:
                    del self._recommendations[recommendation_id]
                    return True
                return False
        else:
            return self._db_delete(recommendation_id)
    
    def _db_delete(self, recommendation_id: str) -> bool:
        """数据库删除"""
        try:
            from backend.schemas.optimization_models import OptimizationRecommendationModel
            db_session = self._db_pool.get_session()
            
            result = db_session.query(OptimizationRecommendationModel).filter_by(id=recommendation_id).delete()
            db_session.commit()
            return result > 0
        except Exception as e:
            logger.error(f"Database delete failed: {e}")
            return False
    
    def list_recommendations(
        self,
        tenant_id: str = None,
        session_id: str = None,
        category: str = None,
        status: str = None,
        priority: str = None,
        limit: int = 100,
        offset: int = 0
    ) -> Tuple[List[Dict[str, Any]], int]:
        """列出建议"""
        if self._use_memory:
            with self._lock:
                recs = list(self._recommendations.values())
                
                # 过滤
                if tenant_id:
                    recs = [r for r in recs if r.get('tenant_id') == tenant_id]
                if session_id:
                    recs = [r for r in recs if r.get('session_id') == session_id]
                if category:
                    recs = [r for r in recs if r.get('category') == category]
                if status:
                    recs = [r for r in recs if r.get('status') == status]
                if priority:
                    recs = [r for r in recs if r.get('priority') == priority]
                
                # 排序：按优先级和创建时间
                priority_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}
                recs.sort(key=lambda x: (
                    priority_order.get(x.get('priority', 'medium'), 2),
                    x.get('created_at', '')
                ), reverse=False)
                
                total = len(recs)
                return [r.copy() for r in recs[offset:offset + limit]], total
        else:
            return self._db_list_recommendations(tenant_id, session_id, category, status, priority, limit, offset)
    
    def _db_list_recommendations(
        self,
        tenant_id: str,
        session_id: str,
        category: str,
        status: str,
        priority: str,
        limit: int,
        offset: int
    ) -> Tuple[List[Dict[str, Any]], int]:
        """数据库列出"""
        try:
            from backend.schemas.optimization_models import OptimizationRecommendationModel
            db_session = self._db_pool.get_session()
            
            query = db_session.query(OptimizationRecommendationModel)
            
            if tenant_id:
                query = query.filter_by(tenant_id=tenant_id)
            if session_id:
                query = query.filter_by(session_id=session_id)
            if category:
                query = query.filter_by(category=category)
            if status:
                query = query.filter_by(status=status)
            if priority:
                query = query.filter_by(priority=priority)
            
            total = query.count()
            models = query.order_by(
                OptimizationRecommendationModel.priority,
                OptimizationRecommendationModel.created_at.desc()
            ).offset(offset).limit(limit).all()
            
            return [m.to_dict() for m in models], total
        except Exception as e:
            logger.error(f"Database list failed: {e}")
            return [], 0
    
    def get_pending_count(self, tenant_id: str = None) -> int:
        """获取待处理建议数量"""
        if self._use_memory:
            with self._lock:
                recs = [r for r in self._recommendations.values() if r.get('status') == 'pending']
                if tenant_id:
                    recs = [r for r in recs if r.get('tenant_id') == tenant_id]
                return len(recs)
        else:
            try:
                from backend.schemas.optimization_models import OptimizationRecommendationModel, RecommendationStatusEnum
                db_session = self._db_pool.get_session()
                
                query = db_session.query(OptimizationRecommendationModel).filter_by(
                    status=RecommendationStatusEnum.PENDING
                )
                if tenant_id:
                    query = query.filter_by(tenant_id=tenant_id)
                
                return query.count()
            except Exception as e:
                logger.error(f"Get pending count failed: {e}")
                return 0
    
    def batch_create(self, recommendations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """批量创建建议"""
        created = []
        for rec in recommendations:
            created.append(self.create(rec))
        return created
    
    def mark_expired(self, before_date: datetime, tenant_id: str = None) -> int:
        """标记过期的建议"""
        count = 0
        if self._use_memory:
            with self._lock:
                for rec in self._recommendations.values():
                    if rec.get('status') == 'pending':
                        created_at = rec.get('created_at')
                        if created_at:
                            try:
                                dt = datetime.fromisoformat(created_at.replace('Z', ''))
                                if dt < before_date:
                                    if not tenant_id or rec.get('tenant_id') == tenant_id:
                                        rec['status'] = 'expired'
                                        count += 1
                            except (ValueError, TypeError):
                                pass
        return count


# ==================== 资源指标快照仓库 ====================

class ResourceMetricSnapshotRepository:
    """资源指标快照仓库
    
    管理资源指标历史数据
    """
    
    def __init__(self, use_memory_storage: bool = True, max_snapshots: int = 10000):
        self._use_memory = use_memory_storage
        self._lock = threading.RLock()
        self._max_snapshots = max_snapshots
        
        # 内存存储 - 按类型分组
        self._snapshots: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        
        if not use_memory_storage:
            self._init_database()
    
    def _init_database(self):
        """初始化数据库连接"""
        try:
            from backend.database.db_pool import DatabasePool
            self._db_pool = DatabasePool()
            logger.info("ResourceMetricSnapshotRepository database initialized")
        except Exception as e:
            logger.warning(f"Failed to init database, using memory: {e}")
            self._use_memory = True
    
    def create(self, snapshot: Dict[str, Any]) -> Dict[str, Any]:
        """创建指标快照"""
        if not snapshot.get('id'):
            snapshot['id'] = f"metric_{uuid.uuid4().hex[:12]}"
        
        if not snapshot.get('timestamp'):
            snapshot['timestamp'] = datetime.utcnow().isoformat()
        
        metric_type = snapshot.get('metric_type', 'system')
        
        if self._use_memory:
            with self._lock:
                self._snapshots[metric_type].append(snapshot.copy())
                # 限制数量
                if len(self._snapshots[metric_type]) > self._max_snapshots:
                    self._snapshots[metric_type] = self._snapshots[metric_type][-self._max_snapshots:]
        else:
            self._db_create(snapshot)
        
        return snapshot
    
    def _db_create(self, snapshot: Dict[str, Any]):
        """数据库创建"""
        try:
            from backend.schemas.optimization_models import ResourceMetricSnapshotModel
            db_session = self._db_pool.get_session()
            
            model = ResourceMetricSnapshotModel(
                id=snapshot['id'],
                tenant_id=snapshot.get('tenant_id'),
                metric_type=snapshot.get('metric_type'),
                metric_name=snapshot.get('metric_name'),
                metric_value=snapshot.get('metric_value'),
                metric_unit=snapshot.get('metric_unit'),
                status=snapshot.get('status', 'normal'),
                threshold_warning=snapshot.get('threshold_warning'),
                threshold_critical=snapshot.get('threshold_critical'),
                source=snapshot.get('source'),
                extra_data=snapshot.get('extra_data', {})
            )
            
            db_session.add(model)
            db_session.commit()
        except Exception as e:
            logger.error(f"Database create failed: {e}")
            metric_type = snapshot.get('metric_type', 'system')
            with self._lock:
                self._snapshots[metric_type].append(snapshot.copy())
    
    def get_history(
        self,
        metric_type: str = None,
        metric_name: str = None,
        tenant_id: str = None,
        start_time: datetime = None,
        end_time: datetime = None,
        limit: int = 1000
    ) -> List[Dict[str, Any]]:
        """获取历史数据"""
        if self._use_memory:
            with self._lock:
                if metric_type:
                    snapshots = self._snapshots.get(metric_type, []).copy()
                else:
                    snapshots = []
                    for type_snapshots in self._snapshots.values():
                        snapshots.extend(type_snapshots)
                
                # 过滤
                if metric_name:
                    snapshots = [s for s in snapshots if s.get('metric_name') == metric_name]
                if tenant_id:
                    snapshots = [s for s in snapshots if s.get('tenant_id') == tenant_id]
                if start_time:
                    snapshots = [s for s in snapshots if s.get('timestamp', '') >= start_time.isoformat()]
                if end_time:
                    snapshots = [s for s in snapshots if s.get('timestamp', '') <= end_time.isoformat()]
                
                # 排序并限制
                snapshots.sort(key=lambda x: x.get('timestamp', ''))
                return snapshots[-limit:]
        else:
            return self._db_get_history(metric_type, metric_name, tenant_id, start_time, end_time, limit)
    
    def _db_get_history(
        self,
        metric_type: str,
        metric_name: str,
        tenant_id: str,
        start_time: datetime,
        end_time: datetime,
        limit: int
    ) -> List[Dict[str, Any]]:
        """数据库获取历史"""
        try:
            from backend.schemas.optimization_models import ResourceMetricSnapshotModel
            db_session = self._db_pool.get_session()
            
            query = db_session.query(ResourceMetricSnapshotModel)
            
            if metric_type:
                query = query.filter_by(metric_type=metric_type)
            if metric_name:
                query = query.filter_by(metric_name=metric_name)
            if tenant_id:
                query = query.filter_by(tenant_id=tenant_id)
            if start_time:
                query = query.filter(ResourceMetricSnapshotModel.timestamp >= start_time)
            if end_time:
                query = query.filter(ResourceMetricSnapshotModel.timestamp <= end_time)
            
            models = query.order_by(ResourceMetricSnapshotModel.timestamp.desc()).limit(limit).all()
            return [m.to_dict() for m in reversed(models)]
        except Exception as e:
            logger.error(f"Database get history failed: {e}")
            return []
    
    def get_latest(self, metric_type: str, metric_name: str = None, tenant_id: str = None) -> Optional[Dict[str, Any]]:
        """获取最新快照"""
        if self._use_memory:
            with self._lock:
                snapshots = self._snapshots.get(metric_type, [])
                if metric_name:
                    snapshots = [s for s in snapshots if s.get('metric_name') == metric_name]
                if tenant_id:
                    snapshots = [s for s in snapshots if s.get('tenant_id') == tenant_id]
                return snapshots[-1].copy() if snapshots else None
        else:
            try:
                from backend.schemas.optimization_models import ResourceMetricSnapshotModel
                db_session = self._db_pool.get_session()
                
                query = db_session.query(ResourceMetricSnapshotModel).filter_by(metric_type=metric_type)
                if metric_name:
                    query = query.filter_by(metric_name=metric_name)
                if tenant_id:
                    query = query.filter_by(tenant_id=tenant_id)
                
                model = query.order_by(ResourceMetricSnapshotModel.timestamp.desc()).first()
                return model.to_dict() if model else None
            except Exception as e:
                logger.error(f"Get latest failed: {e}")
                return None
    
    def cleanup_old_snapshots(self, before_date: datetime, tenant_id: str = None) -> int:
        """清理旧快照"""
        count = 0
        if self._use_memory:
            with self._lock:
                for metric_type in list(self._snapshots.keys()):
                    original_count = len(self._snapshots[metric_type])
                    self._snapshots[metric_type] = [
                        s for s in self._snapshots[metric_type]
                        if s.get('timestamp', '') >= before_date.isoformat()
                        and (not tenant_id or s.get('tenant_id') == tenant_id)
                    ]
                    count += original_count - len(self._snapshots[metric_type])
        else:
            try:
                from backend.schemas.optimization_models import ResourceMetricSnapshotModel
                db_session = self._db_pool.get_session()
                
                query = db_session.query(ResourceMetricSnapshotModel).filter(
                    ResourceMetricSnapshotModel.timestamp < before_date
                )
                if tenant_id:
                    query = query.filter_by(tenant_id=tenant_id)
                
                count = query.delete()
                db_session.commit()
            except Exception as e:
                logger.error(f"Cleanup failed: {e}")
        
        logger.info(f"Cleaned up {count} old snapshots")
        return count
    
    def batch_create(self, snapshots: List[Dict[str, Any]]) -> int:
        """批量创建快照"""
        count = 0
        for snapshot in snapshots:
            self.create(snapshot)
            count += 1
        return count


# ==================== 性能分析报告仓库 ====================

class PerformanceAnalysisReportRepository:
    """性能分析报告仓库
    
    管理性能分析报告的存储和查询
    """
    
    def __init__(self, use_memory_storage: bool = True):
        self._use_memory = use_memory_storage
        self._lock = threading.RLock()
        
        # 内存存储
        self._reports: Dict[str, Dict[str, Any]] = {}
        
        if not use_memory_storage:
            self._init_database()
    
    def _init_database(self):
        """初始化数据库连接"""
        try:
            from backend.database.db_pool import DatabasePool
            self._db_pool = DatabasePool()
            logger.info("PerformanceAnalysisReportRepository database initialized")
        except Exception as e:
            logger.warning(f"Failed to init database, using memory: {e}")
            self._use_memory = True
    
    def create(self, report: Dict[str, Any]) -> Dict[str, Any]:
        """创建报告"""
        if not report.get('id'):
            report['id'] = f"report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        
        if not report.get('created_at'):
            report['created_at'] = datetime.utcnow().isoformat()
        
        if self._use_memory:
            with self._lock:
                self._reports[report['id']] = report.copy()
        else:
            self._db_create(report)
        
        logger.info(f"Created analysis report: {report['id']}")
        return report
    
    def _db_create(self, report: Dict[str, Any]):
        """数据库创建"""
        try:
            from backend.schemas.optimization_models import PerformanceAnalysisReportModel
            db_session = self._db_pool.get_session()
            
            model = PerformanceAnalysisReportModel(
                id=report['id'],
                tenant_id=report.get('tenant_id'),
                analysis_type=report.get('analysis_type', 'full'),
                target_id=report.get('target_id'),
                summary=report.get('summary'),
                bottlenecks=report.get('bottlenecks', []),
                recommendations=report.get('recommendations', []),
                metrics_summary=report.get('metrics_summary', {}),
                status=report.get('status', 'pending'),
                created_by=report.get('created_by')
            )
            
            db_session.add(model)
            db_session.commit()
        except Exception as e:
            logger.error(f"Database create failed: {e}")
            with self._lock:
                self._reports[report['id']] = report.copy()
    
    def get_by_id(self, report_id: str) -> Optional[Dict[str, Any]]:
        """根据ID获取报告"""
        if self._use_memory:
            with self._lock:
                return self._reports.get(report_id, {}).copy() if report_id in self._reports else None
        else:
            return self._db_get_by_id(report_id)
    
    def _db_get_by_id(self, report_id: str) -> Optional[Dict[str, Any]]:
        """数据库获取"""
        try:
            from backend.schemas.optimization_models import PerformanceAnalysisReportModel
            db_session = self._db_pool.get_session()
            
            model = db_session.query(PerformanceAnalysisReportModel).filter_by(id=report_id).first()
            return model.to_dict() if model else None
        except Exception as e:
            logger.error(f"Database get failed: {e}")
            with self._lock:
                return self._reports.get(report_id, {}).copy() if report_id in self._reports else None
    
    def update(self, report_id: str, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """更新报告"""
        if self._use_memory:
            with self._lock:
                if report_id not in self._reports:
                    return None
                self._reports[report_id].update(updates)
                return self._reports[report_id].copy()
        else:
            return self._db_update(report_id, updates)
    
    def _db_update(self, report_id: str, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """数据库更新"""
        try:
            from backend.schemas.optimization_models import PerformanceAnalysisReportModel
            db_session = self._db_pool.get_session()
            
            model = db_session.query(PerformanceAnalysisReportModel).filter_by(id=report_id).first()
            if not model:
                return None
            
            for key, value in updates.items():
                if hasattr(model, key):
                    setattr(model, key, value)
            
            db_session.commit()
            return model.to_dict()
        except Exception as e:
            logger.error(f"Database update failed: {e}")
            return None
    
    def list_reports(
        self,
        tenant_id: str = None,
        analysis_type: str = None,
        status: str = None,
        limit: int = 50,
        offset: int = 0
    ) -> Tuple[List[Dict[str, Any]], int]:
        """列出报告"""
        if self._use_memory:
            with self._lock:
                reports = list(self._reports.values())
                
                if tenant_id:
                    reports = [r for r in reports if r.get('tenant_id') == tenant_id]
                if analysis_type:
                    reports = [r for r in reports if r.get('analysis_type') == analysis_type]
                if status:
                    reports = [r for r in reports if r.get('status') == status]
                
                reports.sort(key=lambda x: x.get('created_at', ''), reverse=True)
                
                total = len(reports)
                return [r.copy() for r in reports[offset:offset + limit]], total
        else:
            return self._db_list_reports(tenant_id, analysis_type, status, limit, offset)
    
    def _db_list_reports(
        self,
        tenant_id: str,
        analysis_type: str,
        status: str,
        limit: int,
        offset: int
    ) -> Tuple[List[Dict[str, Any]], int]:
        """数据库列出"""
        try:
            from backend.schemas.optimization_models import PerformanceAnalysisReportModel
            db_session = self._db_pool.get_session()
            
            query = db_session.query(PerformanceAnalysisReportModel)
            
            if tenant_id:
                query = query.filter_by(tenant_id=tenant_id)
            if analysis_type:
                query = query.filter_by(analysis_type=analysis_type)
            if status:
                query = query.filter_by(status=status)
            
            total = query.count()
            models = query.order_by(PerformanceAnalysisReportModel.created_at.desc()).offset(offset).limit(limit).all()
            
            return [m.to_dict() for m in models], total
        except Exception as e:
            logger.error(f"Database list failed: {e}")
            return [], 0


# ==================== 资源告警仓库 ====================

class ResourceAlertRepository:
    """资源告警仓库
    
    管理资源告警的存储和查询
    """
    
    def __init__(self, use_memory_storage: bool = True):
        self._use_memory = use_memory_storage
        self._lock = threading.RLock()
        
        # 内存存储
        self._alerts: Dict[str, Dict[str, Any]] = {}
        
        if not use_memory_storage:
            self._init_database()
    
    def _init_database(self):
        """初始化数据库连接"""
        try:
            from backend.database.db_pool import DatabasePool
            self._db_pool = DatabasePool()
            logger.info("ResourceAlertRepository database initialized")
        except Exception as e:
            logger.warning(f"Failed to init database, using memory: {e}")
            self._use_memory = True
    
    def create(self, alert: Dict[str, Any]) -> Dict[str, Any]:
        """创建告警"""
        if not alert.get('id'):
            alert['id'] = f"alert_{uuid.uuid4().hex[:12]}"
        
        if not alert.get('timestamp'):
            alert['timestamp'] = datetime.utcnow().isoformat()
        
        if self._use_memory:
            with self._lock:
                self._alerts[alert['id']] = alert.copy()
        else:
            self._db_create(alert)
        
        logger.info(f"Created alert: {alert['id']} - {alert.get('level')} - {alert.get('message')}")
        return alert
    
    def _db_create(self, alert: Dict[str, Any]):
        """数据库创建"""
        try:
            from backend.schemas.optimization_models import ResourceAlertModel
            db_session = self._db_pool.get_session()
            
            model = ResourceAlertModel(
                id=alert['id'],
                tenant_id=alert.get('tenant_id'),
                level=alert.get('level', 'info'),
                resource_type=alert.get('resource_type'),
                message=alert.get('message'),
                metric_name=alert.get('metric_name'),
                metric_value=alert.get('metric_value'),
                threshold=alert.get('threshold'),
                status=alert.get('status', 'active')
            )
            
            db_session.add(model)
            db_session.commit()
        except Exception as e:
            logger.error(f"Database create failed: {e}")
            with self._lock:
                self._alerts[alert['id']] = alert.copy()
    
    def get_by_id(self, alert_id: str) -> Optional[Dict[str, Any]]:
        """根据ID获取告警"""
        if self._use_memory:
            with self._lock:
                return self._alerts.get(alert_id, {}).copy() if alert_id in self._alerts else None
        else:
            return self._db_get_by_id(alert_id)
    
    def _db_get_by_id(self, alert_id: str) -> Optional[Dict[str, Any]]:
        """数据库获取"""
        try:
            from backend.schemas.optimization_models import ResourceAlertModel
            db_session = self._db_pool.get_session()
            
            model = db_session.query(ResourceAlertModel).filter_by(id=alert_id).first()
            return model.to_dict() if model else None
        except Exception as e:
            logger.error(f"Database get failed: {e}")
            with self._lock:
                return self._alerts.get(alert_id, {}).copy() if alert_id in self._alerts else None
    
    def update(self, alert_id: str, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """更新告警"""
        if self._use_memory:
            with self._lock:
                if alert_id not in self._alerts:
                    return None
                self._alerts[alert_id].update(updates)
                return self._alerts[alert_id].copy()
        else:
            return self._db_update(alert_id, updates)
    
    def _db_update(self, alert_id: str, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """数据库更新"""
        try:
            from backend.schemas.optimization_models import ResourceAlertModel
            db_session = self._db_pool.get_session()
            
            model = db_session.query(ResourceAlertModel).filter_by(id=alert_id).first()
            if not model:
                return None
            
            for key, value in updates.items():
                if hasattr(model, key):
                    setattr(model, key, value)
            
            db_session.commit()
            return model.to_dict()
        except Exception as e:
            logger.error(f"Database update failed: {e}")
            return None
    
    def list_alerts(
        self,
        tenant_id: str = None,
        level: str = None,
        resource_type: str = None,
        status: str = None,
        limit: int = 100,
        offset: int = 0
    ) -> Tuple[List[Dict[str, Any]], int]:
        """列出告警"""
        if self._use_memory:
            with self._lock:
                alerts = list(self._alerts.values())
                
                if tenant_id:
                    alerts = [a for a in alerts if a.get('tenant_id') == tenant_id]
                if level:
                    alerts = [a for a in alerts if a.get('level') == level]
                if resource_type:
                    alerts = [a for a in alerts if a.get('resource_type') == resource_type]
                if status:
                    alerts = [a for a in alerts if a.get('status') == status]
                
                # 排序：按级别和时间
                level_order = {'critical': 0, 'warning': 1, 'info': 2}
                alerts.sort(key=lambda x: (
                    level_order.get(x.get('level', 'info'), 2),
                    x.get('timestamp', '')
                ), reverse=True)
                
                total = len(alerts)
                return [a.copy() for a in alerts[offset:offset + limit]], total
        else:
            return self._db_list_alerts(tenant_id, level, resource_type, status, limit, offset)
    
    def _db_list_alerts(
        self,
        tenant_id: str,
        level: str,
        resource_type: str,
        status: str,
        limit: int,
        offset: int
    ) -> Tuple[List[Dict[str, Any]], int]:
        """数据库列出"""
        try:
            from backend.schemas.optimization_models import ResourceAlertModel
            db_session = self._db_pool.get_session()
            
            query = db_session.query(ResourceAlertModel)
            
            if tenant_id:
                query = query.filter_by(tenant_id=tenant_id)
            if level:
                query = query.filter_by(level=level)
            if resource_type:
                query = query.filter_by(resource_type=resource_type)
            if status:
                query = query.filter_by(status=status)
            
            total = query.count()
            models = query.order_by(
                ResourceAlertModel.level,
                ResourceAlertModel.timestamp.desc()
            ).offset(offset).limit(limit).all()
            
            return [m.to_dict() for m in models], total
        except Exception as e:
            logger.error(f"Database list failed: {e}")
            return [], 0
    
    def get_active_count(self, tenant_id: str = None) -> Dict[str, int]:
        """获取活跃告警统计"""
        if self._use_memory:
            with self._lock:
                alerts = [a for a in self._alerts.values() if a.get('status') == 'active']
                if tenant_id:
                    alerts = [a for a in alerts if a.get('tenant_id') == tenant_id]
                
                counts = {'total': len(alerts), 'critical': 0, 'warning': 0, 'info': 0}
                for alert in alerts:
                    level = alert.get('level', 'info')
                    if level in counts:
                        counts[level] += 1
                
                return counts
        else:
            try:
                from backend.schemas.optimization_models import ResourceAlertModel
                from sqlalchemy import func
                db_session = self._db_pool.get_session()
                
                query = db_session.query(
                    ResourceAlertModel.level,
                    func.count(ResourceAlertModel.id)
                ).filter_by(status='active')
                
                if tenant_id:
                    query = query.filter_by(tenant_id=tenant_id)
                
                results = query.group_by(ResourceAlertModel.level).all()
                
                counts = {'total': 0, 'critical': 0, 'warning': 0, 'info': 0}
                for level, count in results:
                    counts[level] = count
                    counts['total'] += count
                
                return counts
            except Exception as e:
                logger.error(f"Get active count failed: {e}")
                return {'total': 0, 'critical': 0, 'warning': 0, 'info': 0}
    
    def acknowledge(self, alert_id: str, user_id: str) -> Optional[Dict[str, Any]]:
        """确认告警"""
        return self.update(alert_id, {
            'status': 'acknowledged',
            'acknowledged_at': datetime.utcnow().isoformat(),
            'acknowledged_by': user_id
        })
    
    def resolve(self, alert_id: str, user_id: str, note: str = None) -> Optional[Dict[str, Any]]:
        """解决告警"""
        updates = {
            'status': 'resolved',
            'resolved_at': datetime.utcnow().isoformat(),
            'resolved_by': user_id
        }
        if note:
            updates['resolution_note'] = note
        return self.update(alert_id, updates)


# ==================== 仓库工厂函数 ====================

_session_repo = None
_recommendation_repo = None
_metric_repo = None
_report_repo = None
_alert_repo = None


def get_optimization_session_repository(use_memory: bool = True) -> OptimizationSessionRepository:
    """获取优化会话仓库实例"""
    global _session_repo
    if _session_repo is None:
        _session_repo = OptimizationSessionRepository(use_memory_storage=use_memory)
    return _session_repo


def get_optimization_recommendation_repository(use_memory: bool = True) -> OptimizationRecommendationRepository:
    """获取优化建议仓库实例"""
    global _recommendation_repo
    if _recommendation_repo is None:
        _recommendation_repo = OptimizationRecommendationRepository(use_memory_storage=use_memory)
    return _recommendation_repo


def get_resource_metric_snapshot_repository(use_memory: bool = True) -> ResourceMetricSnapshotRepository:
    """获取资源指标快照仓库实例"""
    global _metric_repo
    if _metric_repo is None:
        _metric_repo = ResourceMetricSnapshotRepository(use_memory_storage=use_memory)
    return _metric_repo


def get_performance_analysis_report_repository(use_memory: bool = True) -> PerformanceAnalysisReportRepository:
    """获取性能分析报告仓库实例"""
    global _report_repo
    if _report_repo is None:
        _report_repo = PerformanceAnalysisReportRepository(use_memory_storage=use_memory)
    return _report_repo


def get_resource_alert_repository(use_memory: bool = True) -> ResourceAlertRepository:
    """获取资源告警仓库实例"""
    global _alert_repo
    if _alert_repo is None:
        _alert_repo = ResourceAlertRepository(use_memory_storage=use_memory)
    return _alert_repo


# ==================== 导出 ====================

__all__ = [
    # 仓库类
    'OptimizationSessionRepository',
    'OptimizationRecommendationRepository',
    'ResourceMetricSnapshotRepository',
    'PerformanceAnalysisReportRepository',
    'ResourceAlertRepository',
    
    # 工厂函数
    'get_optimization_session_repository',
    'get_optimization_recommendation_repository',
    'get_resource_metric_snapshot_repository',
    'get_performance_analysis_report_repository',
    'get_resource_alert_repository',
]

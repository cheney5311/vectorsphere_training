#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""性能模块仓库层

提供异步任务、性能指标、告警等数据的持久化存储。
支持内存存储和数据库存储两种模式。
"""

import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from collections import deque

logger = logging.getLogger(__name__)


class AsyncTaskRepository:
    """异步任务仓库"""
    
    def __init__(self, use_memory: bool = True, db_session=None):
        self._use_memory = use_memory
        self._db_session = db_session
        self._memory_store: Dict[str, Dict] = {}
        self._max_history = 10000
    
    def _generate_id(self) -> str:
        return f"task_{uuid.uuid4().hex[:12]}"
    
    def create(self, task_data: Dict[str, Any]) -> Optional[Dict]:
        """创建任务"""
        try:
            task_id = task_data.get('id') or self._generate_id()
            now = datetime.utcnow()
            
            task = {
                'id': task_id,
                'tenant_id': task_data.get('tenant_id'),
                'name': task_data.get('name'),
                'category': task_data.get('category'),
                'description': task_data.get('description'),
                'status': task_data.get('status', 'pending'),
                'priority': task_data.get('priority', 'normal'),
                'params': task_data.get('params', {}),
                'result': task_data.get('result'),
                'error_message': task_data.get('error_message'),
                'error_traceback': task_data.get('error_traceback'),
                'created_at': task_data.get('created_at', now),
                'started_at': task_data.get('started_at'),
                'completed_at': task_data.get('completed_at'),
                'timeout': task_data.get('timeout'),
                'execution_time': task_data.get('execution_time'),
                'retry_count': task_data.get('retry_count', 0),
                'max_retries': task_data.get('max_retries', 3),
                'queue_position': task_data.get('queue_position'),
                'worker_id': task_data.get('worker_id'),
                'metadata': task_data.get('metadata', {}),
                'created_by': task_data.get('created_by')
            }
            
            if self._use_memory:
                self._memory_store[task_id] = task
                # 限制历史记录大小
                self._cleanup_old_tasks()
                return task
            else:
                return self._create_db(task)
                
        except Exception as e:
            logger.error(f"Failed to create task: {e}")
            return None
    
    def _create_db(self, task: Dict) -> Optional[Dict]:
        """数据库创建"""
        try:
            from backend.schemas.performance_models import AsyncTaskModel, TaskStatusEnum, TaskPriorityEnum
            
            model = AsyncTaskModel(
                id=task['id'],
                tenant_id=task['tenant_id'],
                name=task['name'],
                category=task['category'],
                description=task['description'],
                status=TaskStatusEnum(task['status']) if task['status'] else TaskStatusEnum.PENDING,
                priority=TaskPriorityEnum(task['priority']) if task['priority'] else TaskPriorityEnum.NORMAL,
                params=task['params'],
                result=task['result'],
                error_message=task['error_message'],
                created_at=task['created_at'],
                started_at=task['started_at'],
                completed_at=task['completed_at'],
                timeout=task['timeout'],
                execution_time=task['execution_time'],
                retry_count=task['retry_count'],
                queue_position=task['queue_position'],
                worker_id=task['worker_id'],
                task_metadata=task['metadata'],
                created_by=task['created_by']
            )
            
            self._db_session.add(model)
            self._db_session.commit()
            return model.to_dict()
            
        except Exception as e:
            self._db_session.rollback()
            logger.error(f"DB create task failed: {e}")
            return None
    
    def get_by_id(self, task_id: str, tenant_id: str = None) -> Optional[Dict]:
        """根据 ID 获取任务"""
        try:
            if self._use_memory:
                task = self._memory_store.get(task_id)
                if task and (tenant_id is None or task.get('tenant_id') == tenant_id):
                    return task
                return None
            else:
                from backend.schemas.performance_models import AsyncTaskModel
                query = self._db_session.query(AsyncTaskModel).filter_by(id=task_id)
                if tenant_id:
                    query = query.filter_by(tenant_id=tenant_id)
                model = query.first()
                return model.to_dict() if model else None
                
        except Exception as e:
            logger.error(f"Failed to get task {task_id}: {e}")
            return None
    
    def get_all(
        self,
        tenant_id: str = None,
        status: str = None,
        category: str = None,
        priority: str = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict]:
        """获取任务列表"""
        try:
            if self._use_memory:
                tasks = list(self._memory_store.values())
                
                if tenant_id:
                    tasks = [t for t in tasks if t.get('tenant_id') == tenant_id]
                if status:
                    tasks = [t for t in tasks if t.get('status') == status]
                if category:
                    tasks = [t for t in tasks if t.get('category') == category]
                if priority:
                    tasks = [t for t in tasks if t.get('priority') == priority]
                
                tasks.sort(key=lambda x: x.get('created_at', datetime.min), reverse=True)
                return tasks[offset:offset + limit]
            else:
                from backend.schemas.performance_models import AsyncTaskModel
                query = self._db_session.query(AsyncTaskModel)
                
                if tenant_id:
                    query = query.filter_by(tenant_id=tenant_id)
                if status:
                    query = query.filter_by(status=status)
                if category:
                    query = query.filter_by(category=category)
                if priority:
                    query = query.filter_by(priority=priority)
                
                query = query.order_by(AsyncTaskModel.created_at.desc())
                query = query.offset(offset).limit(limit)
                
                return [m.to_dict() for m in query.all()]
                
        except Exception as e:
            logger.error(f"Failed to get tasks: {e}")
            return []
    
    def update(self, task_id: str, updates: Dict[str, Any], tenant_id: str = None) -> bool:
        """更新任务"""
        try:
            if self._use_memory:
                if task_id not in self._memory_store:
                    return False
                task = self._memory_store[task_id]
                if tenant_id and task.get('tenant_id') != tenant_id:
                    return False
                task.update(updates)
                return True
            else:
                from backend.schemas.performance_models import AsyncTaskModel
                query = self._db_session.query(AsyncTaskModel).filter_by(id=task_id)
                if tenant_id:
                    query = query.filter_by(tenant_id=tenant_id)
                count = query.update(updates)
                self._db_session.commit()
                return count > 0
                
        except Exception as e:
            logger.error(f"Failed to update task {task_id}: {e}")
            return False
    
    def update_status(
        self,
        task_id: str,
        status: str,
        result: Any = None,
        error_message: str = None,
        execution_time: float = None
    ) -> bool:
        """更新任务状态"""
        updates = {'status': status}
        
        if status == 'running':
            updates['started_at'] = datetime.utcnow()
        elif status in ('completed', 'failed', 'timeout'):
            updates['completed_at'] = datetime.utcnow()
        
        if result is not None:
            updates['result'] = result
        if error_message:
            updates['error_message'] = error_message
        if execution_time is not None:
            updates['execution_time'] = execution_time
        
        return self.update(task_id, updates)
    
    def delete(self, task_id: str, tenant_id: str = None) -> bool:
        """删除任务"""
        try:
            if self._use_memory:
                if task_id in self._memory_store:
                    task = self._memory_store[task_id]
                    if tenant_id and task.get('tenant_id') != tenant_id:
                        return False
                    del self._memory_store[task_id]
                    return True
                return False
            else:
                from backend.schemas.performance_models import AsyncTaskModel
                query = self._db_session.query(AsyncTaskModel).filter_by(id=task_id)
                if tenant_id:
                    query = query.filter_by(tenant_id=tenant_id)
                count = query.delete()
                self._db_session.commit()
                return count > 0
                
        except Exception as e:
            logger.error(f"Failed to delete task {task_id}: {e}")
            return False
    
    def cleanup_old_tasks(self, max_age_seconds: int = 86400 * 7) -> int:
        """清理旧任务"""
        try:
            cutoff = datetime.utcnow() - timedelta(seconds=max_age_seconds)
            count = 0
            
            if self._use_memory:
                to_delete = []
                for task_id, task in self._memory_store.items():
                    created_at = task.get('created_at')
                    if created_at and created_at < cutoff:
                        status = task.get('status')
                        if status in ('completed', 'failed', 'timeout', 'cancelled'):
                            to_delete.append(task_id)
                
                for task_id in to_delete:
                    del self._memory_store[task_id]
                    count += 1
            else:
                from backend.schemas.performance_models import AsyncTaskModel
                count = self._db_session.query(AsyncTaskModel).filter(
                    AsyncTaskModel.created_at < cutoff,
                    AsyncTaskModel.status.in_(['completed', 'failed', 'timeout', 'cancelled'])
                ).delete(synchronize_session=False)
                self._db_session.commit()
            
            return count
            
        except Exception as e:
            logger.error(f"Failed to cleanup old tasks: {e}")
            return 0
    
    def _cleanup_old_tasks(self):
        """内存存储限制清理"""
        if len(self._memory_store) > self._max_history:
            # 按创建时间排序，删除最旧的任务
            sorted_tasks = sorted(
                self._memory_store.items(),
                key=lambda x: x[1].get('created_at', datetime.min)
            )
            to_remove = len(self._memory_store) - self._max_history
            for task_id, _ in sorted_tasks[:to_remove]:
                del self._memory_store[task_id]
    
    def get_statistics(self, tenant_id: str = None) -> Dict[str, Any]:
        """获取任务统计"""
        tasks = self.get_all(tenant_id=tenant_id, limit=100000)
        
        by_status = {}
        by_category = {}
        total_execution_time = 0
        execution_count = 0
        
        for task in tasks:
            status = task.get('status', 'unknown')
            category = task.get('category', 'general')
            
            by_status[status] = by_status.get(status, 0) + 1
            by_category[category] = by_category.get(category, 0) + 1
            
            exec_time = task.get('execution_time')
            if exec_time:
                total_execution_time += exec_time
                execution_count += 1
        
        return {
            'total': len(tasks),
            'by_status': by_status,
            'by_category': by_category,
            'avg_execution_time': total_execution_time / execution_count if execution_count > 0 else 0
        }


class PerformanceMetricRepository:
    """性能指标仓库"""
    
    def __init__(self, use_memory: bool = True, db_session=None):
        self._use_memory = use_memory
        self._db_session = db_session
        self._memory_store: deque = deque(maxlen=100000)
    
    def _generate_id(self) -> str:
        return f"metric_{uuid.uuid4().hex[:12]}"
    
    def record(self, metric_data: Dict[str, Any]) -> Optional[Dict]:
        """记录指标"""
        try:
            metric_id = self._generate_id()
            now = datetime.utcnow()
            
            metric = {
                'id': metric_id,
                'tenant_id': metric_data.get('tenant_id'),
                'metric_type': metric_data.get('metric_type'),
                'metric_name': metric_data.get('metric_name'),
                'metric_value': metric_data.get('metric_value'),
                'metric_unit': metric_data.get('metric_unit'),
                'timestamp': metric_data.get('timestamp', now),
                'period_type': metric_data.get('period_type'),
                'resource_id': metric_data.get('resource_id'),
                'resource_type': metric_data.get('resource_type'),
                'min_value': metric_data.get('min_value'),
                'max_value': metric_data.get('max_value'),
                'avg_value': metric_data.get('avg_value'),
                'sample_count': metric_data.get('sample_count', 1),
                'tags': metric_data.get('tags', {})
            }
            
            if self._use_memory:
                self._memory_store.append(metric)
                return metric
            else:
                return self._record_db(metric)
                
        except Exception as e:
            logger.error(f"Failed to record metric: {e}")
            return None
    
    def _record_db(self, metric: Dict) -> Optional[Dict]:
        """数据库记录"""
        try:
            from backend.schemas.performance_models import PerformanceMetricModel, MetricTypeEnum, ResourceTypeEnum
            
            model = PerformanceMetricModel(
                id=metric['id'],
                tenant_id=metric['tenant_id'],
                metric_type=MetricTypeEnum(metric['metric_type']) if metric['metric_type'] else None,
                metric_name=metric['metric_name'],
                metric_value=metric['metric_value'],
                metric_unit=metric['metric_unit'],
                timestamp=metric['timestamp'],
                period_type=metric['period_type'],
                resource_id=metric['resource_id'],
                resource_type=ResourceTypeEnum(metric['resource_type']) if metric['resource_type'] else None,
                min_value=metric['min_value'],
                max_value=metric['max_value'],
                avg_value=metric['avg_value'],
                sample_count=metric['sample_count'],
                tags=metric['tags']
            )
            
            self._db_session.add(model)
            self._db_session.commit()
            return model.to_dict()
            
        except Exception as e:
            self._db_session.rollback()
            logger.error(f"DB record metric failed: {e}")
            return None
    
    def get_history(
        self,
        metric_type: str = None,
        metric_name: str = None,
        resource_id: str = None,
        start_time: datetime = None,
        end_time: datetime = None,
        limit: int = 1000
    ) -> List[Dict]:
        """获取指标历史"""
        try:
            if self._use_memory:
                metrics = list(self._memory_store)
                
                if metric_type:
                    metrics = [m for m in metrics if m.get('metric_type') == metric_type]
                if metric_name:
                    metrics = [m for m in metrics if m.get('metric_name') == metric_name]
                if resource_id:
                    metrics = [m for m in metrics if m.get('resource_id') == resource_id]
                if start_time:
                    metrics = [m for m in metrics if m.get('timestamp') and m['timestamp'] >= start_time]
                if end_time:
                    metrics = [m for m in metrics if m.get('timestamp') and m['timestamp'] <= end_time]
                
                metrics.sort(key=lambda x: x.get('timestamp', datetime.min), reverse=True)
                return metrics[:limit]
            else:
                from backend.schemas.performance_models import PerformanceMetricModel
                query = self._db_session.query(PerformanceMetricModel)
                
                if metric_type:
                    query = query.filter_by(metric_type=metric_type)
                if metric_name:
                    query = query.filter_by(metric_name=metric_name)
                if resource_id:
                    query = query.filter_by(resource_id=resource_id)
                if start_time:
                    query = query.filter(PerformanceMetricModel.timestamp >= start_time)
                if end_time:
                    query = query.filter(PerformanceMetricModel.timestamp <= end_time)
                
                query = query.order_by(PerformanceMetricModel.timestamp.desc())
                query = query.limit(limit)
                
                return [m.to_dict() for m in query.all()]
                
        except Exception as e:
            logger.error(f"Failed to get metric history: {e}")
            return []
    
    def get_aggregated(
        self,
        metric_type: str,
        metric_name: str,
        period: str = 'hour',
        start_time: datetime = None,
        end_time: datetime = None
    ) -> List[Dict]:
        """获取聚合指标"""
        # 简化实现：获取原始数据并手动聚合
        metrics = self.get_history(
            metric_type=metric_type,
            metric_name=metric_name,
            start_time=start_time,
            end_time=end_time,
            limit=10000
        )
        
        if not metrics:
            return []
        
        # 按时间段聚合
        aggregated = {}
        for m in metrics:
            ts = m.get('timestamp')
            if not ts:
                continue
            
            if isinstance(ts, str):
                ts = datetime.fromisoformat(ts.replace('Z', '+00:00'))
            
            if period == 'minute':
                key = ts.replace(second=0, microsecond=0)
            elif period == 'hour':
                key = ts.replace(minute=0, second=0, microsecond=0)
            elif period == 'day':
                key = ts.replace(hour=0, minute=0, second=0, microsecond=0)
            else:
                key = ts
            
            if key not in aggregated:
                aggregated[key] = {
                    'values': [],
                    'timestamp': key
                }
            aggregated[key]['values'].append(m.get('metric_value', 0))
        
        result = []
        for key, data in sorted(aggregated.items()):
            values = data['values']
            result.append({
                'timestamp': data['timestamp'].isoformat(),
                'avg_value': sum(values) / len(values),
                'min_value': min(values),
                'max_value': max(values),
                'sample_count': len(values)
            })
        
        return result


class AlertRepository:
    """告警仓库"""
    
    def __init__(self, use_memory: bool = True, db_session=None):
        self._use_memory = use_memory
        self._db_session = db_session
        self._memory_store: Dict[str, Dict] = {}
    
    def _generate_id(self) -> str:
        return f"alert_{uuid.uuid4().hex[:12]}"
    
    def create(self, alert_data: Dict[str, Any]) -> Optional[Dict]:
        """创建告警"""
        try:
            alert_id = alert_data.get('id') or self._generate_id()
            now = datetime.utcnow()
            
            alert = {
                'id': alert_id,
                'tenant_id': alert_data.get('tenant_id'),
                'name': alert_data.get('name'),
                'description': alert_data.get('description'),
                'rule_id': alert_data.get('rule_id'),
                'status': alert_data.get('status', 'active'),
                'level': alert_data.get('level', 'medium'),
                'metric_type': alert_data.get('metric_type'),
                'metric_name': alert_data.get('metric_name'),
                'metric_value': alert_data.get('metric_value'),
                'threshold': alert_data.get('threshold'),
                'triggered_at': alert_data.get('triggered_at', now),
                'acknowledged_at': alert_data.get('acknowledged_at'),
                'resolved_at': alert_data.get('resolved_at'),
                'duration_seconds': alert_data.get('duration_seconds'),
                'acknowledged_by': alert_data.get('acknowledged_by'),
                'resolved_by': alert_data.get('resolved_by'),
                'resolution_notes': alert_data.get('resolution_notes'),
                'notification_sent': alert_data.get('notification_sent', False),
                'notification_channels': alert_data.get('notification_channels', []),
                'metadata': alert_data.get('metadata', {})
            }
            
            if self._use_memory:
                self._memory_store[alert_id] = alert
                return alert
            else:
                return self._create_db(alert)
                
        except Exception as e:
            logger.error(f"Failed to create alert: {e}")
            return None
    
    def _create_db(self, alert: Dict) -> Optional[Dict]:
        """数据库创建"""
        try:
            from backend.schemas.performance_models import AlertModel, AlertStatusEnum, AlertLevelEnum, MetricTypeEnum
            
            model = AlertModel(
                id=alert['id'],
                tenant_id=alert['tenant_id'],
                name=alert['name'],
                description=alert['description'],
                rule_id=alert['rule_id'],
                status=AlertStatusEnum(alert['status']) if alert['status'] else AlertStatusEnum.ACTIVE,
                level=AlertLevelEnum(alert['level']) if alert['level'] else AlertLevelEnum.MEDIUM,
                metric_type=MetricTypeEnum(alert['metric_type']) if alert['metric_type'] else None,
                metric_name=alert['metric_name'],
                metric_value=alert['metric_value'],
                threshold=alert['threshold'],
                triggered_at=alert['triggered_at'],
                acknowledged_at=alert['acknowledged_at'],
                resolved_at=alert['resolved_at'],
                duration_seconds=alert['duration_seconds'],
                acknowledged_by=alert['acknowledged_by'],
                resolved_by=alert['resolved_by'],
                resolution_notes=alert['resolution_notes'],
                notification_sent=alert['notification_sent'],
                notification_channels=alert['notification_channels'],
                alert_metadata=alert['metadata']
            )
            
            self._db_session.add(model)
            self._db_session.commit()
            return model.to_dict()
            
        except Exception as e:
            self._db_session.rollback()
            logger.error(f"DB create alert failed: {e}")
            return None
    
    def get_by_id(self, alert_id: str, tenant_id: str = None) -> Optional[Dict]:
        """根据 ID 获取告警"""
        try:
            if self._use_memory:
                alert = self._memory_store.get(alert_id)
                if alert and (tenant_id is None or alert.get('tenant_id') == tenant_id):
                    return alert
                return None
            else:
                from backend.schemas.performance_models import AlertModel
                query = self._db_session.query(AlertModel).filter_by(id=alert_id)
                if tenant_id:
                    query = query.filter_by(tenant_id=tenant_id)
                model = query.first()
                return model.to_dict() if model else None
                
        except Exception as e:
            logger.error(f"Failed to get alert {alert_id}: {e}")
            return None
    
    def get_active(self, tenant_id: str = None, level: str = None) -> List[Dict]:
        """获取活跃告警"""
        return self.get_all(tenant_id=tenant_id, status='active', level=level)
    
    def get_all(
        self,
        tenant_id: str = None,
        status: str = None,
        level: str = None,
        rule_id: str = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict]:
        """获取告警列表"""
        try:
            if self._use_memory:
                alerts = list(self._memory_store.values())
                
                if tenant_id:
                    alerts = [a for a in alerts if a.get('tenant_id') == tenant_id]
                if status:
                    alerts = [a for a in alerts if a.get('status') == status]
                if level:
                    alerts = [a for a in alerts if a.get('level') == level]
                if rule_id:
                    alerts = [a for a in alerts if a.get('rule_id') == rule_id]
                
                alerts.sort(key=lambda x: x.get('triggered_at', datetime.min), reverse=True)
                return alerts[offset:offset + limit]
            else:
                from backend.schemas.performance_models import AlertModel
                query = self._db_session.query(AlertModel)
                
                if tenant_id:
                    query = query.filter_by(tenant_id=tenant_id)
                if status:
                    query = query.filter_by(status=status)
                if level:
                    query = query.filter_by(level=level)
                if rule_id:
                    query = query.filter_by(rule_id=rule_id)
                
                query = query.order_by(AlertModel.triggered_at.desc())
                query = query.offset(offset).limit(limit)
                
                return [m.to_dict() for m in query.all()]
                
        except Exception as e:
            logger.error(f"Failed to get alerts: {e}")
            return []
    
    def update(self, alert_id: str, updates: Dict[str, Any], tenant_id: str = None) -> bool:
        """更新告警"""
        try:
            if self._use_memory:
                if alert_id not in self._memory_store:
                    return False
                alert = self._memory_store[alert_id]
                if tenant_id and alert.get('tenant_id') != tenant_id:
                    return False
                alert.update(updates)
                return True
            else:
                from backend.schemas.performance_models import AlertModel
                query = self._db_session.query(AlertModel).filter_by(id=alert_id)
                if tenant_id:
                    query = query.filter_by(tenant_id=tenant_id)
                count = query.update(updates)
                self._db_session.commit()
                return count > 0
                
        except Exception as e:
            logger.error(f"Failed to update alert {alert_id}: {e}")
            return False
    
    def acknowledge(self, alert_id: str, user_id: str, tenant_id: str = None) -> bool:
        """确认告警"""
        return self.update(alert_id, {
            'status': 'acknowledged',
            'acknowledged_at': datetime.utcnow(),
            'acknowledged_by': user_id
        }, tenant_id)
    
    def resolve(
        self,
        alert_id: str,
        user_id: str,
        notes: str = None,
        tenant_id: str = None
    ) -> bool:
        """解决告警"""
        alert = self.get_by_id(alert_id, tenant_id)
        if not alert:
            return False
        
        triggered_at = alert.get('triggered_at')
        now = datetime.utcnow()
        duration = None
        
        if triggered_at:
            if isinstance(triggered_at, str):
                triggered_at = datetime.fromisoformat(triggered_at.replace('Z', '+00:00'))
            duration = (now - triggered_at).total_seconds()
        
        return self.update(alert_id, {
            'status': 'resolved',
            'resolved_at': now,
            'resolved_by': user_id,
            'resolution_notes': notes,
            'duration_seconds': duration
        }, tenant_id)
    
    def get_statistics(self, tenant_id: str = None) -> Dict[str, Any]:
        """获取告警统计"""
        alerts = self.get_all(tenant_id=tenant_id, limit=100000)
        
        by_status = {}
        by_level = {}
        
        for alert in alerts:
            status = alert.get('status', 'unknown')
            level = alert.get('level', 'unknown')
            
            by_status[status] = by_status.get(status, 0) + 1
            by_level[level] = by_level.get(level, 0) + 1
        
        return {
            'total': len(alerts),
            'active': by_status.get('active', 0),
            'by_status': by_status,
            'by_level': by_level
        }


class AlertRuleRepository:
    """告警规则仓库"""
    
    def __init__(self, use_memory: bool = True, db_session=None):
        self._use_memory = use_memory
        self._db_session = db_session
        self._memory_store: Dict[str, Dict] = {}
    
    def _generate_id(self) -> str:
        return f"rule_{uuid.uuid4().hex[:12]}"
    
    def create(self, rule_data: Dict[str, Any]) -> Optional[Dict]:
        """创建规则"""
        try:
            rule_id = rule_data.get('id') or self._generate_id()
            now = datetime.utcnow()
            
            rule = {
                'id': rule_id,
                'tenant_id': rule_data.get('tenant_id'),
                'name': rule_data.get('name'),
                'description': rule_data.get('description'),
                'metric_type': rule_data.get('metric_type'),
                'metric_name': rule_data.get('metric_name'),
                'operator': rule_data.get('operator'),
                'threshold': rule_data.get('threshold'),
                'duration': rule_data.get('duration', 0),
                'severity': rule_data.get('severity', 'medium'),
                'enabled': rule_data.get('enabled', True),
                'notification_channels': rule_data.get('notification_channels', []),
                'cooldown_seconds': rule_data.get('cooldown_seconds', 300),
                'metadata': rule_data.get('metadata', {}),
                'created_at': now,
                'updated_at': now,
                'created_by': rule_data.get('created_by')
            }
            
            if self._use_memory:
                self._memory_store[rule_id] = rule
                return rule
            else:
                return self._create_db(rule)
                
        except Exception as e:
            logger.error(f"Failed to create rule: {e}")
            return None
    
    def _create_db(self, rule: Dict) -> Optional[Dict]:
        """数据库创建"""
        try:
            from backend.schemas.performance_models import AlertRuleModel, MetricTypeEnum, AlertLevelEnum
            
            model = AlertRuleModel(
                id=rule['id'],
                tenant_id=rule['tenant_id'],
                name=rule['name'],
                description=rule['description'],
                metric_type=MetricTypeEnum(rule['metric_type']) if rule['metric_type'] else None,
                metric_name=rule['metric_name'],
                operator=rule['operator'],
                threshold=rule['threshold'],
                duration=rule['duration'],
                severity=AlertLevelEnum(rule['severity']) if rule['severity'] else AlertLevelEnum.MEDIUM,
                enabled=rule['enabled'],
                notification_channels=rule['notification_channels'],
                cooldown_seconds=rule['cooldown_seconds'],
                rule_metadata=rule['metadata'],
                created_at=rule['created_at'],
                updated_at=rule['updated_at'],
                created_by=rule['created_by']
            )
            
            self._db_session.add(model)
            self._db_session.commit()
            return model.to_dict()
            
        except Exception as e:
            self._db_session.rollback()
            logger.error(f"DB create rule failed: {e}")
            return None
    
    def get_by_id(self, rule_id: str, tenant_id: str = None) -> Optional[Dict]:
        """根据 ID 获取规则"""
        try:
            if self._use_memory:
                rule = self._memory_store.get(rule_id)
                if rule and (tenant_id is None or rule.get('tenant_id') == tenant_id):
                    return rule
                return None
            else:
                from backend.schemas.performance_models import AlertRuleModel
                query = self._db_session.query(AlertRuleModel).filter_by(id=rule_id)
                if tenant_id:
                    query = query.filter_by(tenant_id=tenant_id)
                model = query.first()
                return model.to_dict() if model else None
                
        except Exception as e:
            logger.error(f"Failed to get rule {rule_id}: {e}")
            return None
    
    def get_enabled(self, tenant_id: str = None) -> List[Dict]:
        """获取启用的规则"""
        return self.get_all(tenant_id=tenant_id, enabled=True)
    
    def get_all(
        self,
        tenant_id: str = None,
        enabled: bool = None,
        metric_type: str = None,
        limit: int = 100
    ) -> List[Dict]:
        """获取规则列表"""
        try:
            if self._use_memory:
                rules = list(self._memory_store.values())
                
                if tenant_id:
                    rules = [r for r in rules if r.get('tenant_id') == tenant_id]
                if enabled is not None:
                    rules = [r for r in rules if r.get('enabled') == enabled]
                if metric_type:
                    rules = [r for r in rules if r.get('metric_type') == metric_type]
                
                rules.sort(key=lambda x: x.get('created_at', datetime.min), reverse=True)
                return rules[:limit]
            else:
                from backend.schemas.performance_models import AlertRuleModel
                query = self._db_session.query(AlertRuleModel)
                
                if tenant_id:
                    query = query.filter_by(tenant_id=tenant_id)
                if enabled is not None:
                    query = query.filter_by(enabled=enabled)
                if metric_type:
                    query = query.filter_by(metric_type=metric_type)
                
                query = query.order_by(AlertRuleModel.created_at.desc())
                query = query.limit(limit)
                
                return [m.to_dict() for m in query.all()]
                
        except Exception as e:
            logger.error(f"Failed to get rules: {e}")
            return []
    
    def update(self, rule_id: str, updates: Dict[str, Any], tenant_id: str = None) -> bool:
        """更新规则"""
        try:
            updates['updated_at'] = datetime.utcnow()
            
            if self._use_memory:
                if rule_id not in self._memory_store:
                    return False
                rule = self._memory_store[rule_id]
                if tenant_id and rule.get('tenant_id') != tenant_id:
                    return False
                rule.update(updates)
                return True
            else:
                from backend.schemas.performance_models import AlertRuleModel
                query = self._db_session.query(AlertRuleModel).filter_by(id=rule_id)
                if tenant_id:
                    query = query.filter_by(tenant_id=tenant_id)
                count = query.update(updates)
                self._db_session.commit()
                return count > 0
                
        except Exception as e:
            logger.error(f"Failed to update rule {rule_id}: {e}")
            return False
    
    def toggle_enabled(self, rule_id: str, enabled: bool, tenant_id: str = None) -> bool:
        """切换规则启用状态"""
        return self.update(rule_id, {'enabled': enabled}, tenant_id)
    
    def delete(self, rule_id: str, tenant_id: str = None) -> bool:
        """删除规则"""
        try:
            if self._use_memory:
                if rule_id in self._memory_store:
                    rule = self._memory_store[rule_id]
                    if tenant_id and rule.get('tenant_id') != tenant_id:
                        return False
                    del self._memory_store[rule_id]
                    return True
                return False
            else:
                from backend.schemas.performance_models import AlertRuleModel
                query = self._db_session.query(AlertRuleModel).filter_by(id=rule_id)
                if tenant_id:
                    query = query.filter_by(tenant_id=tenant_id)
                count = query.delete()
                self._db_session.commit()
                return count > 0
                
        except Exception as e:
            logger.error(f"Failed to delete rule {rule_id}: {e}")
            return False


class SystemSnapshotRepository:
    """系统快照仓库"""
    
    def __init__(self, use_memory: bool = True, db_session=None):
        self._use_memory = use_memory
        self._db_session = db_session
        self._memory_store: deque = deque(maxlen=10000)
    
    def _generate_id(self) -> str:
        return f"snap_{uuid.uuid4().hex[:12]}"
    
    def record(self, snapshot_data: Dict[str, Any]) -> Optional[Dict]:
        """记录快照"""
        try:
            snapshot_id = self._generate_id()
            now = datetime.utcnow()
            
            snapshot = {
                'id': snapshot_id,
                'tenant_id': snapshot_data.get('tenant_id'),
                'timestamp': snapshot_data.get('timestamp', now),
                'cpu_percent': snapshot_data.get('cpu_percent', 0),
                'cpu_count': snapshot_data.get('cpu_count', 0),
                'load_average_1m': snapshot_data.get('load_average_1m', 0),
                'load_average_5m': snapshot_data.get('load_average_5m', 0),
                'load_average_15m': snapshot_data.get('load_average_15m', 0),
                'memory_percent': snapshot_data.get('memory_percent', 0),
                'memory_total_gb': snapshot_data.get('memory_total_gb', 0),
                'memory_used_gb': snapshot_data.get('memory_used_gb', 0),
                'memory_available_gb': snapshot_data.get('memory_available_gb', 0),
                'disk_percent': snapshot_data.get('disk_percent', 0),
                'disk_total_gb': snapshot_data.get('disk_total_gb', 0),
                'disk_used_gb': snapshot_data.get('disk_used_gb', 0),
                'disk_free_gb': snapshot_data.get('disk_free_gb', 0),
                'network_bytes_sent': snapshot_data.get('network_bytes_sent', 0),
                'network_bytes_recv': snapshot_data.get('network_bytes_recv', 0),
                'network_connections': snapshot_data.get('network_connections', 0),
                'process_count': snapshot_data.get('process_count', 0),
                'metadata': snapshot_data.get('metadata', {})
            }
            
            if self._use_memory:
                self._memory_store.append(snapshot)
                return snapshot
            else:
                return self._record_db(snapshot)
                
        except Exception as e:
            logger.error(f"Failed to record snapshot: {e}")
            return None
    
    def _record_db(self, snapshot: Dict) -> Optional[Dict]:
        """数据库记录"""
        try:
            from backend.schemas.performance_models import SystemSnapshotModel
            
            model = SystemSnapshotModel(**snapshot)
            self._db_session.add(model)
            self._db_session.commit()
            return model.to_dict()
            
        except Exception as e:
            self._db_session.rollback()
            logger.error(f"DB record snapshot failed: {e}")
            return None
    
    def get_latest(self, tenant_id: str = None) -> Optional[Dict]:
        """获取最新快照"""
        history = self.get_history(tenant_id=tenant_id, limit=1)
        return history[0] if history else None
    
    def get_history(
        self,
        tenant_id: str = None,
        start_time: datetime = None,
        end_time: datetime = None,
        limit: int = 100
    ) -> List[Dict]:
        """获取快照历史"""
        try:
            if self._use_memory:
                snapshots = list(self._memory_store)
                
                if tenant_id:
                    snapshots = [s for s in snapshots if s.get('tenant_id') == tenant_id]
                if start_time:
                    snapshots = [s for s in snapshots if s.get('timestamp') and s['timestamp'] >= start_time]
                if end_time:
                    snapshots = [s for s in snapshots if s.get('timestamp') and s['timestamp'] <= end_time]
                
                snapshots.sort(key=lambda x: x.get('timestamp', datetime.min), reverse=True)
                return snapshots[:limit]
            else:
                from backend.schemas.performance_models import SystemSnapshotModel
                query = self._db_session.query(SystemSnapshotModel)
                
                if tenant_id:
                    query = query.filter_by(tenant_id=tenant_id)
                if start_time:
                    query = query.filter(SystemSnapshotModel.timestamp >= start_time)
                if end_time:
                    query = query.filter(SystemSnapshotModel.timestamp <= end_time)
                
                query = query.order_by(SystemSnapshotModel.timestamp.desc())
                query = query.limit(limit)
                
                return [m.to_dict() for m in query.all()]
                
        except Exception as e:
            logger.error(f"Failed to get snapshot history: {e}")
            return []


# ==================== 单例获取函数 ====================

_task_repo = None
_metric_repo = None
_alert_repo = None
_rule_repo = None
_snapshot_repo = None


def get_async_task_repository(use_memory: bool = True) -> AsyncTaskRepository:
    """获取任务仓库"""
    global _task_repo
    if _task_repo is None:
        _task_repo = AsyncTaskRepository(use_memory=use_memory)
    return _task_repo


def get_performance_metric_repository(use_memory: bool = True) -> PerformanceMetricRepository:
    """获取指标仓库"""
    global _metric_repo
    if _metric_repo is None:
        _metric_repo = PerformanceMetricRepository(use_memory=use_memory)
    return _metric_repo


def get_alert_repository(use_memory: bool = True) -> AlertRepository:
    """获取告警仓库"""
    global _alert_repo
    if _alert_repo is None:
        _alert_repo = AlertRepository(use_memory=use_memory)
    return _alert_repo


def get_alert_rule_repository(use_memory: bool = True) -> AlertRuleRepository:
    """获取规则仓库"""
    global _rule_repo
    if _rule_repo is None:
        _rule_repo = AlertRuleRepository(use_memory=use_memory)
    return _rule_repo


def get_system_snapshot_repository(use_memory: bool = True) -> SystemSnapshotRepository:
    """获取快照仓库"""
    global _snapshot_repo
    if _snapshot_repo is None:
        _snapshot_repo = SystemSnapshotRepository(use_memory=use_memory)
    return _snapshot_repo


def reset_performance_repositories():
    """重置所有仓库（用于测试）"""
    global _task_repo, _metric_repo, _alert_repo, _rule_repo, _snapshot_repo
    _task_repo = None
    _metric_repo = None
    _alert_repo = None
    _rule_repo = None
    _snapshot_repo = None

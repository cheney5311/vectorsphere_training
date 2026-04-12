"""监控运维数据访问层

提供监控运维相关的数据库访问功能，包括：
- 性能指标记录
- 告警规则管理
- 告警历史记录
- 自动化任务记录
- 监控报告记录
"""

import json
import logging
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime, timedelta
import uuid

logger = logging.getLogger(__name__)


def _generate_id() -> str:
    """生成唯一ID"""
    return str(uuid.uuid4())


# ==============================================================================
# 性能指标仓库
# ==============================================================================

class PerformanceMetricRepository:
    """性能指标数据访问层"""
    
    def __init__(self, use_memory_storage: bool = False):
        """初始化性能指标仓库
        
        Args:
            use_memory_storage: 是否使用内存存储
        """
        self._use_memory_storage = use_memory_storage
        
        if use_memory_storage:
            self._metrics: Dict[str, Dict] = {}
            self._db_manager = None
        else:
            try:
                from backend.modules.database.manager import get_database_manager
                self._db_manager = get_database_manager()
            except ImportError:
                logger.warning("Cannot import database manager, falling back to memory storage")
                self._use_memory_storage = True
                self._metrics: Dict[str, Dict] = {}
                self._db_manager = None
    
    def create(self, metric_data: Dict[str, Any]) -> Dict[str, Any]:
        """创建性能指标记录
        
        Args:
            metric_data: 指标数据
            
        Returns:
            创建的指标记录
        """
        try:
            record_id = metric_data.get('id') or _generate_id()
            metric_id = metric_data.get('metric_id') or f"metric_{_generate_id()[:8]}"
            
            if self._use_memory_storage:
                metric_data['id'] = record_id
                metric_data['metric_id'] = metric_id
                metric_data['created_at'] = datetime.utcnow().isoformat()
                metric_data['updated_at'] = datetime.utcnow().isoformat()
                metric_data.setdefault('timestamp', datetime.utcnow().isoformat())
                self._metrics[record_id] = metric_data
                return metric_data
            
            from backend.schemas.monitoring_models import PerformanceMetricRecord
            
            with self._db_manager.get_db_session() as db:
                # 处理 timestamp
                timestamp = metric_data.get('timestamp')
                if isinstance(timestamp, str):
                    timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                elif not timestamp:
                    timestamp = datetime.utcnow()
                
                metric = PerformanceMetricRecord(
                    id=record_id,
                    tenant_id=metric_data.get('tenant_id'),
                    metric_id=metric_id,
                    deployment_id=metric_data.get('deployment_id'),
                    metric_type=metric_data.get('metric_type'),
                    value=metric_data.get('value'),
                    unit=metric_data.get('unit'),
                    timestamp=timestamp,
                    source=metric_data.get('source', 'system'),
                    tags=metric_data.get('tags'),
                    metadata_=metric_data.get('metadata')
                )
                
                db.add(metric)
                db.commit()
                db.refresh(metric)
                
                return metric.to_dict()
                
        except Exception as e:
            logger.error(f"Failed to create performance metric: {e}")
            raise
    
    def batch_create(self, metrics_data: List[Dict[str, Any]], tenant_id: str) -> List[Dict[str, Any]]:
        """批量创建性能指标记录
        
        Args:
            metrics_data: 指标数据列表
            tenant_id: 租户ID
            
        Returns:
            创建的指标记录列表
        """
        results = []
        for metric_data in metrics_data:
            metric_data['tenant_id'] = tenant_id
            try:
                result = self.create(metric_data)
                results.append(result)
            except Exception as e:
                logger.warning(f"Failed to create metric: {e}")
                continue
        return results
    
    def get_by_id(self, metric_id: str, tenant_id: str) -> Optional[Dict[str, Any]]:
        """通过指标ID获取记录
        
        Args:
            metric_id: 指标ID
            tenant_id: 租户ID
            
        Returns:
            指标记录
        """
        try:
            if self._use_memory_storage:
                for record in self._metrics.values():
                    if record.get('metric_id') == metric_id and record.get('tenant_id') == tenant_id:
                        return record
                return None
            
            from backend.schemas.monitoring_models import PerformanceMetricRecord
            
            with self._db_manager.get_db_session() as db:
                metric = db.query(PerformanceMetricRecord).filter(
                    PerformanceMetricRecord.metric_id == metric_id,
                    PerformanceMetricRecord.tenant_id == tenant_id
                ).first()
                
                return metric.to_dict() if metric else None
                
        except Exception as e:
            logger.error(f"Failed to get metric by ID: {e}")
            return None
    
    def list_by_deployment(self, deployment_id: str, tenant_id: str,
                          metric_types: Optional[List[str]] = None,
                          start_time: Optional[datetime] = None,
                          end_time: Optional[datetime] = None,
                          limit: int = 100,
                          offset: int = 0) -> Tuple[List[Dict[str, Any]], int]:
        """获取部署的性能指标列表
        
        Args:
            deployment_id: 部署ID
            tenant_id: 租户ID
            metric_types: 指标类型过滤
            start_time: 开始时间
            end_time: 结束时间
            limit: 返回数量限制
            offset: 偏移量
            
        Returns:
            (指标列表, 总数)
        """
        try:
            if self._use_memory_storage:
                results = []
                for record in self._metrics.values():
                    if record.get('deployment_id') != deployment_id:
                        continue
                    if record.get('tenant_id') != tenant_id:
                        continue
                    if metric_types and record.get('metric_type') not in metric_types:
                        continue
                    
                    record_time = record.get('timestamp')
                    if isinstance(record_time, str):
                        record_time = datetime.fromisoformat(record_time.replace('Z', '+00:00'))
                    
                    if start_time and record_time < start_time:
                        continue
                    if end_time and record_time > end_time:
                        continue
                    
                    results.append(record)
                
                # 按时间排序
                results.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
                total = len(results)
                return results[offset:offset + limit], total
            
            from backend.schemas.monitoring_models import PerformanceMetricRecord
            
            with self._db_manager.get_db_session() as db:
                query = db.query(PerformanceMetricRecord).filter(
                    PerformanceMetricRecord.deployment_id == deployment_id,
                    PerformanceMetricRecord.tenant_id == tenant_id
                )
                
                if metric_types:
                    query = query.filter(PerformanceMetricRecord.metric_type.in_(metric_types))
                if start_time:
                    query = query.filter(PerformanceMetricRecord.timestamp >= start_time)
                if end_time:
                    query = query.filter(PerformanceMetricRecord.timestamp <= end_time)
                
                total = query.count()
                metrics = query.order_by(PerformanceMetricRecord.timestamp.desc()).offset(offset).limit(limit).all()
                
                return [m.to_dict() for m in metrics], total
                
        except Exception as e:
            logger.error(f"Failed to list metrics by deployment: {e}")
            return [], 0
    
    def get_latest_metrics(self, deployment_id: str, tenant_id: str,
                          metric_types: Optional[List[str]] = None) -> Dict[str, Dict[str, Any]]:
        """获取部署的最新指标值
        
        Args:
            deployment_id: 部署ID
            tenant_id: 租户ID
            metric_types: 指标类型过滤
            
        Returns:
            指标类型到最新值的映射
        """
        try:
            if self._use_memory_storage:
                latest = {}
                for record in self._metrics.values():
                    if record.get('deployment_id') != deployment_id:
                        continue
                    if record.get('tenant_id') != tenant_id:
                        continue
                    
                    m_type = record.get('metric_type')
                    if metric_types and m_type not in metric_types:
                        continue
                    
                    if m_type not in latest:
                        latest[m_type] = record
                    else:
                        current_time = latest[m_type].get('timestamp', '')
                        new_time = record.get('timestamp', '')
                        if new_time > current_time:
                            latest[m_type] = record
                
                return latest
            
            from backend.schemas.monitoring_models import PerformanceMetricRecord
            from sqlalchemy import func
            
            with self._db_manager.get_db_session() as db:
                # 子查询获取每个指标类型的最新时间戳
                subquery = db.query(
                    PerformanceMetricRecord.metric_type,
                    func.max(PerformanceMetricRecord.timestamp).label('max_ts')
                ).filter(
                    PerformanceMetricRecord.deployment_id == deployment_id,
                    PerformanceMetricRecord.tenant_id == tenant_id
                )
                
                if metric_types:
                    subquery = subquery.filter(PerformanceMetricRecord.metric_type.in_(metric_types))
                
                subquery = subquery.group_by(PerformanceMetricRecord.metric_type).subquery()
                
                # 连接获取最新记录
                metrics = db.query(PerformanceMetricRecord).join(
                    subquery,
                    (PerformanceMetricRecord.metric_type == subquery.c.metric_type) &
                    (PerformanceMetricRecord.timestamp == subquery.c.max_ts)
                ).filter(
                    PerformanceMetricRecord.deployment_id == deployment_id,
                    PerformanceMetricRecord.tenant_id == tenant_id
                ).all()
                
                return {m.metric_type: m.to_dict() for m in metrics}
                
        except Exception as e:
            logger.error(f"Failed to get latest metrics: {e}")
            return {}
    
    def get_statistics(self, tenant_id: str, deployment_id: Optional[str] = None,
                      start_time: Optional[datetime] = None,
                      end_time: Optional[datetime] = None) -> Dict[str, Any]:
        """获取指标统计信息
        
        Args:
            tenant_id: 租户ID
            deployment_id: 部署ID（可选）
            start_time: 开始时间
            end_time: 结束时间
            
        Returns:
            统计信息
        """
        try:
            if self._use_memory_storage:
                stats = {
                    'total_records': 0,
                    'deployments': set(),
                    'metric_types': {},
                    'time_range': {'start': None, 'end': None}
                }
                
                for record in self._metrics.values():
                    if record.get('tenant_id') != tenant_id:
                        continue
                    if deployment_id and record.get('deployment_id') != deployment_id:
                        continue
                    
                    record_time = record.get('timestamp')
                    if isinstance(record_time, str):
                        record_time = datetime.fromisoformat(record_time.replace('Z', '+00:00'))
                    
                    if start_time and record_time < start_time:
                        continue
                    if end_time and record_time > end_time:
                        continue
                    
                    stats['total_records'] += 1
                    stats['deployments'].add(record.get('deployment_id'))
                    
                    m_type = record.get('metric_type')
                    if m_type not in stats['metric_types']:
                        stats['metric_types'][m_type] = {
                            'count': 0,
                            'sum': 0,
                            'min': float('inf'),
                            'max': float('-inf')
                        }
                    
                    value = record.get('value', 0)
                    stats['metric_types'][m_type]['count'] += 1
                    stats['metric_types'][m_type]['sum'] += value
                    stats['metric_types'][m_type]['min'] = min(stats['metric_types'][m_type]['min'], value)
                    stats['metric_types'][m_type]['max'] = max(stats['metric_types'][m_type]['max'], value)
                
                # 计算平均值
                for m_type in stats['metric_types']:
                    count = stats['metric_types'][m_type]['count']
                    if count > 0:
                        stats['metric_types'][m_type]['avg'] = stats['metric_types'][m_type]['sum'] / count
                    else:
                        stats['metric_types'][m_type]['avg'] = 0
                
                stats['deployments'] = list(stats['deployments'])
                return stats
            
            from backend.schemas.monitoring_models import PerformanceMetricRecord
            from sqlalchemy import func
            
            with self._db_manager.get_db_session() as db:
                query = db.query(PerformanceMetricRecord).filter(
                    PerformanceMetricRecord.tenant_id == tenant_id
                )
                
                if deployment_id:
                    query = query.filter(PerformanceMetricRecord.deployment_id == deployment_id)
                if start_time:
                    query = query.filter(PerformanceMetricRecord.timestamp >= start_time)
                if end_time:
                    query = query.filter(PerformanceMetricRecord.timestamp <= end_time)
                
                total = query.count()
                
                # 按指标类型聚合
                type_stats = db.query(
                    PerformanceMetricRecord.metric_type,
                    func.count(PerformanceMetricRecord.id).label('count'),
                    func.avg(PerformanceMetricRecord.value).label('avg'),
                    func.min(PerformanceMetricRecord.value).label('min'),
                    func.max(PerformanceMetricRecord.value).label('max')
                ).filter(
                    PerformanceMetricRecord.tenant_id == tenant_id
                )
                
                if deployment_id:
                    type_stats = type_stats.filter(PerformanceMetricRecord.deployment_id == deployment_id)
                if start_time:
                    type_stats = type_stats.filter(PerformanceMetricRecord.timestamp >= start_time)
                if end_time:
                    type_stats = type_stats.filter(PerformanceMetricRecord.timestamp <= end_time)
                
                type_stats = type_stats.group_by(PerformanceMetricRecord.metric_type).all()
                
                # 获取部署列表
                deployments = db.query(PerformanceMetricRecord.deployment_id).filter(
                    PerformanceMetricRecord.tenant_id == tenant_id
                ).distinct().all()
                
                return {
                    'total_records': total,
                    'deployments': [d[0] for d in deployments],
                    'metric_types': {
                        row.metric_type: {
                            'count': row.count,
                            'avg': float(row.avg) if row.avg else 0,
                            'min': float(row.min) if row.min else 0,
                            'max': float(row.max) if row.max else 0
                        }
                        for row in type_stats
                    }
                }
                
        except Exception as e:
            logger.error(f"Failed to get metric statistics: {e}")
            return {'total_records': 0, 'deployments': [], 'metric_types': {}}
    
    def delete_old_records(self, tenant_id: str, retention_days: int = 30) -> int:
        """删除过期的指标记录
        
        Args:
            tenant_id: 租户ID
            retention_days: 保留天数
            
        Returns:
            删除的记录数
        """
        try:
            cutoff_time = datetime.utcnow() - timedelta(days=retention_days)
            
            if self._use_memory_storage:
                to_delete = []
                for record_id, record in self._metrics.items():
                    if record.get('tenant_id') != tenant_id:
                        continue
                    
                    record_time = record.get('timestamp')
                    if isinstance(record_time, str):
                        record_time = datetime.fromisoformat(record_time.replace('Z', '+00:00'))
                    
                    if record_time < cutoff_time:
                        to_delete.append(record_id)
                
                for record_id in to_delete:
                    del self._metrics[record_id]
                
                return len(to_delete)
            
            from backend.schemas.monitoring_models import PerformanceMetricRecord
            
            with self._db_manager.get_db_session() as db:
                deleted = db.query(PerformanceMetricRecord).filter(
                    PerformanceMetricRecord.tenant_id == tenant_id,
                    PerformanceMetricRecord.timestamp < cutoff_time
                ).delete(synchronize_session=False)
                
                db.commit()
                return deleted
                
        except Exception as e:
            logger.error(f"Failed to delete old metric records: {e}")
            return 0


# ==============================================================================
# 告警规则仓库
# ==============================================================================

class AlertRuleRepository:
    """告警规则数据访问层"""
    
    def __init__(self, use_memory_storage: bool = False):
        """初始化告警规则仓库"""
        self._use_memory_storage = use_memory_storage
        
        if use_memory_storage:
            self._rules: Dict[str, Dict] = {}
            self._db_manager = None
        else:
            try:
                from backend.modules.database.manager import get_database_manager
                self._db_manager = get_database_manager()
            except ImportError:
                logger.warning("Cannot import database manager, falling back to memory storage")
                self._use_memory_storage = True
                self._rules: Dict[str, Dict] = {}
                self._db_manager = None
    
    def create(self, rule_data: Dict[str, Any]) -> Dict[str, Any]:
        """创建告警规则"""
        try:
            record_id = rule_data.get('id') or _generate_id()
            rule_id = rule_data.get('rule_id') or f"rule_{_generate_id()[:8]}"
            
            if self._use_memory_storage:
                rule_data['id'] = record_id
                rule_data['rule_id'] = rule_id
                rule_data['created_at'] = datetime.utcnow().isoformat()
                rule_data['updated_at'] = datetime.utcnow().isoformat()
                rule_data.setdefault('enabled', True)
                rule_data.setdefault('trigger_count', 0)
                self._rules[record_id] = rule_data
                return rule_data
            
            from backend.schemas.monitoring_models import AlertRuleRecord
            
            with self._db_manager.get_db_session() as db:
                rule = AlertRuleRecord(
                    id=record_id,
                    tenant_id=rule_data.get('tenant_id'),
                    rule_id=rule_id,
                    name=rule_data.get('name'),
                    description=rule_data.get('description'),
                    metric_type=rule_data.get('metric_type'),
                    threshold=rule_data.get('threshold'),
                    operator=rule_data.get('operator'),
                    severity=rule_data.get('severity'),
                    duration=rule_data.get('duration', 0),
                    deployment_id=rule_data.get('deployment_id'),
                    scope=rule_data.get('scope', 'deployment'),
                    enabled=rule_data.get('enabled', True),
                    notification_channels=rule_data.get('notification_channels'),
                    cooldown_seconds=rule_data.get('cooldown_seconds', 300),
                    created_by=rule_data.get('created_by')
                )
                
                db.add(rule)
                db.commit()
                db.refresh(rule)
                
                return rule.to_dict()
                
        except Exception as e:
            logger.error(f"Failed to create alert rule: {e}")
            raise
    
    def get_by_id(self, rule_id: str, tenant_id: str) -> Optional[Dict[str, Any]]:
        """通过规则ID获取"""
        try:
            if self._use_memory_storage:
                for record in self._rules.values():
                    if record.get('rule_id') == rule_id and record.get('tenant_id') == tenant_id:
                        return record
                return None
            
            from backend.schemas.monitoring_models import AlertRuleRecord
            
            with self._db_manager.get_db_session() as db:
                rule = db.query(AlertRuleRecord).filter(
                    AlertRuleRecord.rule_id == rule_id,
                    AlertRuleRecord.tenant_id == tenant_id
                ).first()
                
                return rule.to_dict() if rule else None
                
        except Exception as e:
            logger.error(f"Failed to get alert rule: {e}")
            return None
    
    def update(self, rule_id: str, tenant_id: str, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """更新告警规则"""
        try:
            if self._use_memory_storage:
                for record_id, record in self._rules.items():
                    if record.get('rule_id') == rule_id and record.get('tenant_id') == tenant_id:
                        record.update(updates)
                        record['updated_at'] = datetime.utcnow().isoformat()
                        return record
                return None
            
            from backend.schemas.monitoring_models import AlertRuleRecord
            
            with self._db_manager.get_db_session() as db:
                rule = db.query(AlertRuleRecord).filter(
                    AlertRuleRecord.rule_id == rule_id,
                    AlertRuleRecord.tenant_id == tenant_id
                ).first()
                
                if not rule:
                    return None
                
                for key, value in updates.items():
                    if hasattr(rule, key):
                        setattr(rule, key, value)
                
                db.commit()
                db.refresh(rule)
                
                return rule.to_dict()
                
        except Exception as e:
            logger.error(f"Failed to update alert rule: {e}")
            return None
    
    def delete(self, rule_id: str, tenant_id: str) -> bool:
        """删除告警规则"""
        try:
            if self._use_memory_storage:
                for record_id, record in list(self._rules.items()):
                    if record.get('rule_id') == rule_id and record.get('tenant_id') == tenant_id:
                        del self._rules[record_id]
                        return True
                return False
            
            from backend.schemas.monitoring_models import AlertRuleRecord
            
            with self._db_manager.get_db_session() as db:
                deleted = db.query(AlertRuleRecord).filter(
                    AlertRuleRecord.rule_id == rule_id,
                    AlertRuleRecord.tenant_id == tenant_id
                ).delete(synchronize_session=False)
                
                db.commit()
                return deleted > 0
                
        except Exception as e:
            logger.error(f"Failed to delete alert rule: {e}")
            return False
    
    def list_by_tenant(self, tenant_id: str, enabled_only: bool = False,
                      deployment_id: Optional[str] = None,
                      limit: int = 100, offset: int = 0) -> Tuple[List[Dict[str, Any]], int]:
        """获取租户的告警规则列表"""
        try:
            if self._use_memory_storage:
                results = []
                for record in self._rules.values():
                    if record.get('tenant_id') != tenant_id:
                        continue
                    if enabled_only and not record.get('enabled', True):
                        continue
                    if deployment_id and record.get('deployment_id') != deployment_id:
                        continue
                    results.append(record)
                
                results.sort(key=lambda x: x.get('created_at', ''), reverse=True)
                total = len(results)
                return results[offset:offset + limit], total
            
            from backend.schemas.monitoring_models import AlertRuleRecord
            
            with self._db_manager.get_db_session() as db:
                query = db.query(AlertRuleRecord).filter(
                    AlertRuleRecord.tenant_id == tenant_id
                )
                
                if enabled_only:
                    query = query.filter(AlertRuleRecord.enabled == True)
                if deployment_id:
                    query = query.filter(AlertRuleRecord.deployment_id == deployment_id)
                
                total = query.count()
                rules = query.order_by(AlertRuleRecord.created_at.desc()).offset(offset).limit(limit).all()
                
                return [r.to_dict() for r in rules], total
                
        except Exception as e:
            logger.error(f"Failed to list alert rules: {e}")
            return [], 0
    
    def increment_trigger_count(self, rule_id: str, tenant_id: str) -> bool:
        """增加规则触发次数"""
        try:
            if self._use_memory_storage:
                for record in self._rules.values():
                    if record.get('rule_id') == rule_id and record.get('tenant_id') == tenant_id:
                        record['trigger_count'] = record.get('trigger_count', 0) + 1
                        record['last_triggered_at'] = datetime.utcnow().isoformat()
                        return True
                return False
            
            from backend.schemas.monitoring_models import AlertRuleRecord
            
            with self._db_manager.get_db_session() as db:
                rule = db.query(AlertRuleRecord).filter(
                    AlertRuleRecord.rule_id == rule_id,
                    AlertRuleRecord.tenant_id == tenant_id
                ).first()
                
                if rule:
                    rule.trigger_count = (rule.trigger_count or 0) + 1
                    rule.last_triggered_at = datetime.utcnow()
                    db.commit()
                    return True
                return False
                
        except Exception as e:
            logger.error(f"Failed to increment trigger count: {e}")
            return False


# ==============================================================================
# 告警历史仓库
# ==============================================================================

class AlertHistoryRepository:
    """告警历史数据访问层"""
    
    def __init__(self, use_memory_storage: bool = False):
        """初始化告警历史仓库"""
        self._use_memory_storage = use_memory_storage
        
        if use_memory_storage:
            self._alerts: Dict[str, Dict] = {}
            self._db_manager = None
        else:
            try:
                from backend.modules.database.manager import get_database_manager
                self._db_manager = get_database_manager()
            except ImportError:
                logger.warning("Cannot import database manager, falling back to memory storage")
                self._use_memory_storage = True
                self._alerts: Dict[str, Dict] = {}
                self._db_manager = None
    
    def create(self, alert_data: Dict[str, Any]) -> Dict[str, Any]:
        """创建告警记录"""
        try:
            record_id = alert_data.get('id') or _generate_id()
            alert_id = alert_data.get('alert_id') or f"alert_{_generate_id()[:8]}"
            
            if self._use_memory_storage:
                alert_data['id'] = record_id
                alert_data['alert_id'] = alert_id
                alert_data['created_at'] = datetime.utcnow().isoformat()
                alert_data['updated_at'] = datetime.utcnow().isoformat()
                alert_data.setdefault('triggered_at', datetime.utcnow().isoformat())
                alert_data.setdefault('resolved', False)
                alert_data.setdefault('acknowledged', False)
                self._alerts[record_id] = alert_data
                return alert_data
            
            from backend.schemas.monitoring_models import AlertHistoryRecord
            
            with self._db_manager.get_db_session() as db:
                # 处理时间字段
                triggered_at = alert_data.get('triggered_at')
                if isinstance(triggered_at, str):
                    triggered_at = datetime.fromisoformat(triggered_at.replace('Z', '+00:00'))
                elif not triggered_at:
                    triggered_at = datetime.utcnow()
                
                alert = AlertHistoryRecord(
                    id=record_id,
                    tenant_id=alert_data.get('tenant_id'),
                    alert_id=alert_id,
                    rule_id=alert_data.get('rule_id'),
                    rule_name=alert_data.get('rule_name'),
                    deployment_id=alert_data.get('deployment_id'),
                    severity=alert_data.get('severity'),
                    message=alert_data.get('message'),
                    metric_type=alert_data.get('metric_type'),
                    metric_value=alert_data.get('metric_value'),
                    threshold=alert_data.get('threshold'),
                    triggered_at=triggered_at,
                    resolved=alert_data.get('resolved', False),
                    acknowledged=alert_data.get('acknowledged', False),
                    context=alert_data.get('context'),
                    metadata_=alert_data.get('metadata')
                )
                
                db.add(alert)
                db.commit()
                db.refresh(alert)
                
                return alert.to_dict()
                
        except Exception as e:
            logger.error(f"Failed to create alert history: {e}")
            raise
    
    def get_by_id(self, alert_id: str, tenant_id: str) -> Optional[Dict[str, Any]]:
        """通过告警ID获取"""
        try:
            if self._use_memory_storage:
                for record in self._alerts.values():
                    if record.get('alert_id') == alert_id and record.get('tenant_id') == tenant_id:
                        return record
                return None
            
            from backend.schemas.monitoring_models import AlertHistoryRecord
            
            with self._db_manager.get_db_session() as db:
                alert = db.query(AlertHistoryRecord).filter(
                    AlertHistoryRecord.alert_id == alert_id,
                    AlertHistoryRecord.tenant_id == tenant_id
                ).first()
                
                return alert.to_dict() if alert else None
                
        except Exception as e:
            logger.error(f"Failed to get alert history: {e}")
            return None
    
    def resolve(self, alert_id: str, tenant_id: str, 
               resolution_notes: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """解决告警"""
        try:
            now = datetime.utcnow()
            
            if self._use_memory_storage:
                for record in self._alerts.values():
                    if record.get('alert_id') == alert_id and record.get('tenant_id') == tenant_id:
                        record['resolved'] = True
                        record['resolved_at'] = now.isoformat()
                        if resolution_notes:
                            record['resolution_notes'] = resolution_notes
                        record['updated_at'] = now.isoformat()
                        return record
                return None
            
            from backend.schemas.monitoring_models import AlertHistoryRecord
            
            with self._db_manager.get_db_session() as db:
                alert = db.query(AlertHistoryRecord).filter(
                    AlertHistoryRecord.alert_id == alert_id,
                    AlertHistoryRecord.tenant_id == tenant_id
                ).first()
                
                if not alert:
                    return None
                
                alert.resolved = True
                alert.resolved_at = now
                if resolution_notes:
                    alert.resolution_notes = resolution_notes
                
                db.commit()
                db.refresh(alert)
                
                return alert.to_dict()
                
        except Exception as e:
            logger.error(f"Failed to resolve alert: {e}")
            return None
    
    def acknowledge(self, alert_id: str, tenant_id: str, 
                   user_id: str) -> Optional[Dict[str, Any]]:
        """确认告警"""
        try:
            now = datetime.utcnow()
            
            if self._use_memory_storage:
                for record in self._alerts.values():
                    if record.get('alert_id') == alert_id and record.get('tenant_id') == tenant_id:
                        record['acknowledged'] = True
                        record['acknowledged_at'] = now.isoformat()
                        record['acknowledged_by'] = user_id
                        record['updated_at'] = now.isoformat()
                        return record
                return None
            
            from backend.schemas.monitoring_models import AlertHistoryRecord
            
            with self._db_manager.get_db_session() as db:
                alert = db.query(AlertHistoryRecord).filter(
                    AlertHistoryRecord.alert_id == alert_id,
                    AlertHistoryRecord.tenant_id == tenant_id
                ).first()
                
                if not alert:
                    return None
                
                alert.acknowledged = True
                alert.acknowledged_at = now
                alert.acknowledged_by = user_id
                
                db.commit()
                db.refresh(alert)
                
                return alert.to_dict()
                
        except Exception as e:
            logger.error(f"Failed to acknowledge alert: {e}")
            return None
    
    def list_by_deployment(self, deployment_id: str, tenant_id: str,
                          severity: Optional[str] = None,
                          resolved: Optional[bool] = None,
                          start_time: Optional[datetime] = None,
                          end_time: Optional[datetime] = None,
                          limit: int = 50, offset: int = 0) -> Tuple[List[Dict[str, Any]], int]:
        """获取部署的告警历史"""
        try:
            if self._use_memory_storage:
                results = []
                for record in self._alerts.values():
                    if record.get('deployment_id') != deployment_id:
                        continue
                    if record.get('tenant_id') != tenant_id:
                        continue
                    if severity and record.get('severity') != severity:
                        continue
                    if resolved is not None and record.get('resolved') != resolved:
                        continue
                    
                    record_time = record.get('triggered_at')
                    if isinstance(record_time, str):
                        record_time = datetime.fromisoformat(record_time.replace('Z', '+00:00'))
                    
                    if start_time and record_time < start_time:
                        continue
                    if end_time and record_time > end_time:
                        continue
                    
                    results.append(record)
                
                results.sort(key=lambda x: x.get('triggered_at', ''), reverse=True)
                total = len(results)
                return results[offset:offset + limit], total
            
            from backend.schemas.monitoring_models import AlertHistoryRecord
            
            with self._db_manager.get_db_session() as db:
                query = db.query(AlertHistoryRecord).filter(
                    AlertHistoryRecord.deployment_id == deployment_id,
                    AlertHistoryRecord.tenant_id == tenant_id
                )
                
                if severity:
                    query = query.filter(AlertHistoryRecord.severity == severity)
                if resolved is not None:
                    query = query.filter(AlertHistoryRecord.resolved == resolved)
                if start_time:
                    query = query.filter(AlertHistoryRecord.triggered_at >= start_time)
                if end_time:
                    query = query.filter(AlertHistoryRecord.triggered_at <= end_time)
                
                total = query.count()
                alerts = query.order_by(AlertHistoryRecord.triggered_at.desc()).offset(offset).limit(limit).all()
                
                return [a.to_dict() for a in alerts], total
                
        except Exception as e:
            logger.error(f"Failed to list alerts by deployment: {e}")
            return [], 0
    
    def get_statistics(self, tenant_id: str, deployment_id: Optional[str] = None,
                      start_time: Optional[datetime] = None,
                      end_time: Optional[datetime] = None) -> Dict[str, Any]:
        """获取告警统计信息"""
        try:
            if self._use_memory_storage:
                stats = {
                    'total': 0,
                    'resolved': 0,
                    'unresolved': 0,
                    'acknowledged': 0,
                    'by_severity': {},
                    'by_metric_type': {}
                }
                
                for record in self._alerts.values():
                    if record.get('tenant_id') != tenant_id:
                        continue
                    if deployment_id and record.get('deployment_id') != deployment_id:
                        continue
                    
                    record_time = record.get('triggered_at')
                    if isinstance(record_time, str):
                        record_time = datetime.fromisoformat(record_time.replace('Z', '+00:00'))
                    
                    if start_time and record_time < start_time:
                        continue
                    if end_time and record_time > end_time:
                        continue
                    
                    stats['total'] += 1
                    if record.get('resolved'):
                        stats['resolved'] += 1
                    else:
                        stats['unresolved'] += 1
                    if record.get('acknowledged'):
                        stats['acknowledged'] += 1
                    
                    severity = record.get('severity', 'unknown')
                    stats['by_severity'][severity] = stats['by_severity'].get(severity, 0) + 1
                    
                    metric_type = record.get('metric_type', 'unknown')
                    stats['by_metric_type'][metric_type] = stats['by_metric_type'].get(metric_type, 0) + 1
                
                return stats
            
            from backend.schemas.monitoring_models import AlertHistoryRecord
            from sqlalchemy import func
            
            with self._db_manager.get_db_session() as db:
                query = db.query(AlertHistoryRecord).filter(
                    AlertHistoryRecord.tenant_id == tenant_id
                )
                
                if deployment_id:
                    query = query.filter(AlertHistoryRecord.deployment_id == deployment_id)
                if start_time:
                    query = query.filter(AlertHistoryRecord.triggered_at >= start_time)
                if end_time:
                    query = query.filter(AlertHistoryRecord.triggered_at <= end_time)
                
                total = query.count()
                resolved = query.filter(AlertHistoryRecord.resolved == True).count()
                acknowledged = query.filter(AlertHistoryRecord.acknowledged == True).count()
                
                # 按严重程度统计
                severity_stats = db.query(
                    AlertHistoryRecord.severity,
                    func.count(AlertHistoryRecord.id)
                ).filter(
                    AlertHistoryRecord.tenant_id == tenant_id
                )
                if deployment_id:
                    severity_stats = severity_stats.filter(AlertHistoryRecord.deployment_id == deployment_id)
                if start_time:
                    severity_stats = severity_stats.filter(AlertHistoryRecord.triggered_at >= start_time)
                if end_time:
                    severity_stats = severity_stats.filter(AlertHistoryRecord.triggered_at <= end_time)
                severity_stats = severity_stats.group_by(AlertHistoryRecord.severity).all()
                
                return {
                    'total': total,
                    'resolved': resolved,
                    'unresolved': total - resolved,
                    'acknowledged': acknowledged,
                    'by_severity': {s: c for s, c in severity_stats}
                }
                
        except Exception as e:
            logger.error(f"Failed to get alert statistics: {e}")
            return {'total': 0, 'resolved': 0, 'unresolved': 0, 'acknowledged': 0, 'by_severity': {}}


# ==============================================================================
# 自动化任务仓库
# ==============================================================================

class AutomationTaskRepository:
    """自动化任务数据访问层"""
    
    def __init__(self, use_memory_storage: bool = False):
        """初始化自动化任务仓库"""
        self._use_memory_storage = use_memory_storage
        
        if use_memory_storage:
            self._tasks: Dict[str, Dict] = {}
            self._db_manager = None
        else:
            try:
                from backend.modules.database.manager import get_database_manager
                self._db_manager = get_database_manager()
            except ImportError:
                logger.warning("Cannot import database manager, falling back to memory storage")
                self._use_memory_storage = True
                self._tasks: Dict[str, Dict] = {}
                self._db_manager = None
    
    def create(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """创建自动化任务记录"""
        try:
            record_id = task_data.get('id') or _generate_id()
            task_id = task_data.get('task_id') or f"task_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{_generate_id()[:4]}"
            
            if self._use_memory_storage:
                task_data['id'] = record_id
                task_data['task_id'] = task_id
                task_data['created_at'] = datetime.utcnow().isoformat()
                task_data['updated_at'] = datetime.utcnow().isoformat()
                task_data.setdefault('status', 'pending')
                task_data.setdefault('retry_count', 0)
                self._tasks[record_id] = task_data
                return task_data
            
            from backend.schemas.monitoring_models import AutomationTaskRecord
            
            with self._db_manager.get_db_session() as db:
                task = AutomationTaskRecord(
                    id=record_id,
                    tenant_id=task_data.get('tenant_id'),
                    task_id=task_id,
                    name=task_data.get('name'),
                    description=task_data.get('description'),
                    task_type=task_data.get('task_type'),
                    status=task_data.get('status', 'pending'),
                    deployment_id=task_data.get('deployment_id'),
                    alert_id=task_data.get('alert_id'),
                    parameters=task_data.get('parameters'),
                    executed_by=task_data.get('executed_by', 'system'),
                    user_id=task_data.get('user_id'),
                    priority=task_data.get('priority', 'normal'),
                    max_retries=task_data.get('max_retries', 3),
                    metadata_=task_data.get('metadata')
                )
                
                db.add(task)
                db.commit()
                db.refresh(task)
                
                return task.to_dict()
                
        except Exception as e:
            logger.error(f"Failed to create automation task: {e}")
            raise
    
    def get_by_id(self, task_id: str, tenant_id: str) -> Optional[Dict[str, Any]]:
        """通过任务ID获取"""
        try:
            if self._use_memory_storage:
                for record in self._tasks.values():
                    if record.get('task_id') == task_id and record.get('tenant_id') == tenant_id:
                        return record
                return None
            
            from backend.schemas.monitoring_models import AutomationTaskRecord
            
            with self._db_manager.get_db_session() as db:
                task = db.query(AutomationTaskRecord).filter(
                    AutomationTaskRecord.task_id == task_id,
                    AutomationTaskRecord.tenant_id == tenant_id
                ).first()
                
                return task.to_dict() if task else None
                
        except Exception as e:
            logger.error(f"Failed to get automation task: {e}")
            return None
    
    def update_status(self, task_id: str, tenant_id: str, status: str,
                     result: Optional[Dict] = None,
                     error_message: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """更新任务状态"""
        try:
            now = datetime.utcnow()
            
            if self._use_memory_storage:
                for record in self._tasks.values():
                    if record.get('task_id') == task_id and record.get('tenant_id') == tenant_id:
                        record['status'] = status
                        record['updated_at'] = now.isoformat()
                        
                        if status == 'running' and not record.get('started_at'):
                            record['started_at'] = now.isoformat()
                        elif status in ['completed', 'failed', 'cancelled']:
                            record['completed_at'] = now.isoformat()
                        
                        if result is not None:
                            record['result'] = result
                        if error_message:
                            record['error_message'] = error_message
                        
                        return record
                return None
            
            from backend.schemas.monitoring_models import AutomationTaskRecord
            
            with self._db_manager.get_db_session() as db:
                task = db.query(AutomationTaskRecord).filter(
                    AutomationTaskRecord.task_id == task_id,
                    AutomationTaskRecord.tenant_id == tenant_id
                ).first()
                
                if not task:
                    return None
                
                task.status = status
                
                if status == 'running' and not task.started_at:
                    task.started_at = now
                elif status in ['completed', 'failed', 'cancelled']:
                    task.completed_at = now
                
                if result is not None:
                    task.result = result
                if error_message:
                    task.error_message = error_message
                
                db.commit()
                db.refresh(task)
                
                return task.to_dict()
                
        except Exception as e:
            logger.error(f"Failed to update task status: {e}")
            return None
    
    def increment_retry(self, task_id: str, tenant_id: str) -> Optional[Dict[str, Any]]:
        """增加重试次数"""
        try:
            if self._use_memory_storage:
                for record in self._tasks.values():
                    if record.get('task_id') == task_id and record.get('tenant_id') == tenant_id:
                        record['retry_count'] = record.get('retry_count', 0) + 1
                        record['status'] = 'pending'
                        record['updated_at'] = datetime.utcnow().isoformat()
                        return record
                return None
            
            from backend.schemas.monitoring_models import AutomationTaskRecord
            
            with self._db_manager.get_db_session() as db:
                task = db.query(AutomationTaskRecord).filter(
                    AutomationTaskRecord.task_id == task_id,
                    AutomationTaskRecord.tenant_id == tenant_id
                ).first()
                
                if not task:
                    return None
                
                task.retry_count = (task.retry_count or 0) + 1
                task.status = 'pending'
                
                db.commit()
                db.refresh(task)
                
                return task.to_dict()
                
        except Exception as e:
            logger.error(f"Failed to increment retry: {e}")
            return None
    
    def list_by_tenant(self, tenant_id: str, status: Optional[str] = None,
                      task_type: Optional[str] = None,
                      deployment_id: Optional[str] = None,
                      limit: int = 50, offset: int = 0) -> Tuple[List[Dict[str, Any]], int]:
        """获取租户的任务列表"""
        try:
            if self._use_memory_storage:
                results = []
                for record in self._tasks.values():
                    if record.get('tenant_id') != tenant_id:
                        continue
                    if status and record.get('status') != status:
                        continue
                    if task_type and record.get('task_type') != task_type:
                        continue
                    if deployment_id and record.get('deployment_id') != deployment_id:
                        continue
                    results.append(record)
                
                results.sort(key=lambda x: x.get('created_at', ''), reverse=True)
                total = len(results)
                return results[offset:offset + limit], total
            
            from backend.schemas.monitoring_models import AutomationTaskRecord
            
            with self._db_manager.get_db_session() as db:
                query = db.query(AutomationTaskRecord).filter(
                    AutomationTaskRecord.tenant_id == tenant_id
                )
                
                if status:
                    query = query.filter(AutomationTaskRecord.status == status)
                if task_type:
                    query = query.filter(AutomationTaskRecord.task_type == task_type)
                if deployment_id:
                    query = query.filter(AutomationTaskRecord.deployment_id == deployment_id)
                
                total = query.count()
                tasks = query.order_by(AutomationTaskRecord.created_at.desc()).offset(offset).limit(limit).all()
                
                return [t.to_dict() for t in tasks], total
                
        except Exception as e:
            logger.error(f"Failed to list automation tasks: {e}")
            return [], 0
    
    def get_pending_tasks(self, tenant_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """获取待处理的任务"""
        try:
            if self._use_memory_storage:
                results = []
                for record in self._tasks.values():
                    if tenant_id and record.get('tenant_id') != tenant_id:
                        continue
                    if record.get('status') == 'pending':
                        results.append(record)
                
                results.sort(key=lambda x: (
                    {'critical': 0, 'high': 1, 'normal': 2, 'low': 3}.get(x.get('priority', 'normal'), 2),
                    x.get('created_at', '')
                ))
                return results
            
            from backend.schemas.monitoring_models import AutomationTaskRecord
            
            with self._db_manager.get_db_session() as db:
                query = db.query(AutomationTaskRecord).filter(
                    AutomationTaskRecord.status == 'pending'
                )
                
                if tenant_id:
                    query = query.filter(AutomationTaskRecord.tenant_id == tenant_id)
                
                tasks = query.order_by(
                    AutomationTaskRecord.priority.desc(),
                    AutomationTaskRecord.created_at.asc()
                ).all()
                
                return [t.to_dict() for t in tasks]
                
        except Exception as e:
            logger.error(f"Failed to get pending tasks: {e}")
            return []
    
    def get_statistics(self, tenant_id: str, deployment_id: Optional[str] = None) -> Dict[str, Any]:
        """获取任务统计信息"""
        try:
            if self._use_memory_storage:
                stats = {
                    'total': 0,
                    'by_status': {},
                    'by_type': {},
                    'success_rate': 0.0
                }
                
                completed = 0
                failed = 0
                
                for record in self._tasks.values():
                    if record.get('tenant_id') != tenant_id:
                        continue
                    if deployment_id and record.get('deployment_id') != deployment_id:
                        continue
                    
                    stats['total'] += 1
                    
                    status = record.get('status', 'unknown')
                    stats['by_status'][status] = stats['by_status'].get(status, 0) + 1
                    
                    if status == 'completed':
                        completed += 1
                    elif status == 'failed':
                        failed += 1
                    
                    task_type = record.get('task_type', 'unknown')
                    stats['by_type'][task_type] = stats['by_type'].get(task_type, 0) + 1
                
                if completed + failed > 0:
                    stats['success_rate'] = completed / (completed + failed)
                
                return stats
            
            from backend.schemas.monitoring_models import AutomationTaskRecord
            from sqlalchemy import func
            
            with self._db_manager.get_db_session() as db:
                query = db.query(AutomationTaskRecord).filter(
                    AutomationTaskRecord.tenant_id == tenant_id
                )
                
                if deployment_id:
                    query = query.filter(AutomationTaskRecord.deployment_id == deployment_id)
                
                total = query.count()
                
                # 按状态统计
                status_stats = db.query(
                    AutomationTaskRecord.status,
                    func.count(AutomationTaskRecord.id)
                ).filter(AutomationTaskRecord.tenant_id == tenant_id)
                if deployment_id:
                    status_stats = status_stats.filter(AutomationTaskRecord.deployment_id == deployment_id)
                status_stats = status_stats.group_by(AutomationTaskRecord.status).all()
                
                # 按类型统计
                type_stats = db.query(
                    AutomationTaskRecord.task_type,
                    func.count(AutomationTaskRecord.id)
                ).filter(AutomationTaskRecord.tenant_id == tenant_id)
                if deployment_id:
                    type_stats = type_stats.filter(AutomationTaskRecord.deployment_id == deployment_id)
                type_stats = type_stats.group_by(AutomationTaskRecord.task_type).all()
                
                by_status = {s: c for s, c in status_stats}
                completed = by_status.get('completed', 0)
                failed = by_status.get('failed', 0)
                success_rate = completed / (completed + failed) if (completed + failed) > 0 else 0.0
                
                return {
                    'total': total,
                    'by_status': by_status,
                    'by_type': {t: c for t, c in type_stats},
                    'success_rate': success_rate
                }
                
        except Exception as e:
            logger.error(f"Failed to get task statistics: {e}")
            return {'total': 0, 'by_status': {}, 'by_type': {}, 'success_rate': 0.0}


# ==============================================================================
# 监控报告仓库
# ==============================================================================

class MonitoringReportRepository:
    """监控报告数据访问层"""
    
    def __init__(self, use_memory_storage: bool = False):
        """初始化监控报告仓库"""
        self._use_memory_storage = use_memory_storage
        
        if use_memory_storage:
            self._reports: Dict[str, Dict] = {}
            self._db_manager = None
        else:
            try:
                from backend.modules.database.manager import get_database_manager
                self._db_manager = get_database_manager()
            except ImportError:
                logger.warning("Cannot import database manager, falling back to memory storage")
                self._use_memory_storage = True
                self._reports: Dict[str, Dict] = {}
                self._db_manager = None
    
    def create(self, report_data: Dict[str, Any]) -> Dict[str, Any]:
        """创建监控报告"""
        try:
            record_id = report_data.get('id') or _generate_id()
            report_id = report_data.get('report_id') or f"report_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{_generate_id()[:4]}"
            
            if self._use_memory_storage:
                report_data['id'] = record_id
                report_data['report_id'] = report_id
                report_data['created_at'] = datetime.utcnow().isoformat()
                report_data['updated_at'] = datetime.utcnow().isoformat()
                report_data.setdefault('status', 'completed')
                self._reports[record_id] = report_data
                return report_data
            
            from backend.schemas.monitoring_models import MonitoringReportRecord
            
            with self._db_manager.get_db_session() as db:
                # 处理时间字段
                time_range_start = report_data.get('time_range_start')
                time_range_end = report_data.get('time_range_end')
                
                if isinstance(time_range_start, str):
                    time_range_start = datetime.fromisoformat(time_range_start.replace('Z', '+00:00'))
                if isinstance(time_range_end, str):
                    time_range_end = datetime.fromisoformat(time_range_end.replace('Z', '+00:00'))
                
                report = MonitoringReportRecord(
                    id=record_id,
                    tenant_id=report_data.get('tenant_id'),
                    report_id=report_id,
                    name=report_data.get('name'),
                    report_type=report_data.get('report_type'),
                    deployment_id=report_data.get('deployment_id'),
                    scope=report_data.get('scope', 'deployment'),
                    time_range_start=time_range_start,
                    time_range_end=time_range_end,
                    summary=report_data.get('summary'),
                    metrics_data=report_data.get('metrics_data'),
                    trends=report_data.get('trends'),
                    capacity_analysis=report_data.get('capacity_analysis'),
                    cost_analysis=report_data.get('cost_analysis'),
                    recommendations=report_data.get('recommendations'),
                    generated_by=report_data.get('generated_by', 'system'),
                    user_id=report_data.get('user_id'),
                    status=report_data.get('status', 'completed')
                )
                
                db.add(report)
                db.commit()
                db.refresh(report)
                
                return report.to_dict()
                
        except Exception as e:
            logger.error(f"Failed to create monitoring report: {e}")
            raise
    
    def get_by_id(self, report_id: str, tenant_id: str) -> Optional[Dict[str, Any]]:
        """通过报告ID获取"""
        try:
            if self._use_memory_storage:
                for record in self._reports.values():
                    if record.get('report_id') == report_id and record.get('tenant_id') == tenant_id:
                        return record
                return None
            
            from backend.schemas.monitoring_models import MonitoringReportRecord
            
            with self._db_manager.get_db_session() as db:
                report = db.query(MonitoringReportRecord).filter(
                    MonitoringReportRecord.report_id == report_id,
                    MonitoringReportRecord.tenant_id == tenant_id
                ).first()
                
                return report.to_dict() if report else None
                
        except Exception as e:
            logger.error(f"Failed to get monitoring report: {e}")
            return None
    
    def list_by_tenant(self, tenant_id: str, report_type: Optional[str] = None,
                      deployment_id: Optional[str] = None,
                      limit: int = 20, offset: int = 0) -> Tuple[List[Dict[str, Any]], int]:
        """获取租户的报告列表"""
        try:
            if self._use_memory_storage:
                results = []
                for record in self._reports.values():
                    if record.get('tenant_id') != tenant_id:
                        continue
                    if report_type and record.get('report_type') != report_type:
                        continue
                    if deployment_id and record.get('deployment_id') != deployment_id:
                        continue
                    results.append(record)
                
                results.sort(key=lambda x: x.get('created_at', ''), reverse=True)
                total = len(results)
                return results[offset:offset + limit], total
            
            from backend.schemas.monitoring_models import MonitoringReportRecord
            
            with self._db_manager.get_db_session() as db:
                query = db.query(MonitoringReportRecord).filter(
                    MonitoringReportRecord.tenant_id == tenant_id
                )
                
                if report_type:
                    query = query.filter(MonitoringReportRecord.report_type == report_type)
                if deployment_id:
                    query = query.filter(MonitoringReportRecord.deployment_id == deployment_id)
                
                total = query.count()
                reports = query.order_by(MonitoringReportRecord.created_at.desc()).offset(offset).limit(limit).all()
                
                return [r.to_dict() for r in reports], total
                
        except Exception as e:
            logger.error(f"Failed to list monitoring reports: {e}")
            return [], 0
    
    def delete(self, report_id: str, tenant_id: str) -> bool:
        """删除报告"""
        try:
            if self._use_memory_storage:
                for record_id, record in list(self._reports.items()):
                    if record.get('report_id') == report_id and record.get('tenant_id') == tenant_id:
                        del self._reports[record_id]
                        return True
                return False
            
            from backend.schemas.monitoring_models import MonitoringReportRecord
            
            with self._db_manager.get_db_session() as db:
                deleted = db.query(MonitoringReportRecord).filter(
                    MonitoringReportRecord.report_id == report_id,
                    MonitoringReportRecord.tenant_id == tenant_id
                ).delete(synchronize_session=False)
                
                db.commit()
                return deleted > 0
                
        except Exception as e:
            logger.error(f"Failed to delete monitoring report: {e}")
            return False


# ==============================================================================
# 获取仓库实例的辅助函数
# ==============================================================================

# 全局仓库实例缓存
_performance_metric_repository: Optional[PerformanceMetricRepository] = None
_alert_rule_repository: Optional[AlertRuleRepository] = None
_alert_history_repository: Optional[AlertHistoryRepository] = None
_automation_task_repository: Optional[AutomationTaskRepository] = None
_monitoring_report_repository: Optional[MonitoringReportRepository] = None


def get_performance_metric_repository(use_memory_storage: bool = False) -> PerformanceMetricRepository:
    """获取性能指标仓库实例"""
    global _performance_metric_repository
    if _performance_metric_repository is None:
        _performance_metric_repository = PerformanceMetricRepository(use_memory_storage=use_memory_storage)
    return _performance_metric_repository


def get_alert_rule_repository(use_memory_storage: bool = False) -> AlertRuleRepository:
    """获取告警规则仓库实例"""
    global _alert_rule_repository
    if _alert_rule_repository is None:
        _alert_rule_repository = AlertRuleRepository(use_memory_storage=use_memory_storage)
    return _alert_rule_repository


def get_alert_history_repository(use_memory_storage: bool = False) -> AlertHistoryRepository:
    """获取告警历史仓库实例"""
    global _alert_history_repository
    if _alert_history_repository is None:
        _alert_history_repository = AlertHistoryRepository(use_memory_storage=use_memory_storage)
    return _alert_history_repository


def get_automation_task_repository(use_memory_storage: bool = False) -> AutomationTaskRepository:
    """获取自动化任务仓库实例"""
    global _automation_task_repository
    if _automation_task_repository is None:
        _automation_task_repository = AutomationTaskRepository(use_memory_storage=use_memory_storage)
    return _automation_task_repository


def get_monitoring_report_repository(use_memory_storage: bool = False) -> MonitoringReportRepository:
    """获取监控报告仓库实例"""
    global _monitoring_report_repository
    if _monitoring_report_repository is None:
        _monitoring_report_repository = MonitoringReportRepository(use_memory_storage=use_memory_storage)
    return _monitoring_report_repository


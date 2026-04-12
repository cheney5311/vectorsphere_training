"""租户审计日志数据访问层

提供租户审计日志相关的数据库访问功能。
"""

import json
import logging
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime

from backend.core.exceptions import DatabaseError
from backend.schemas.base_models import TenantAuditLog

logger = logging.getLogger(__name__)


class TenantAuditLogRepository:
    """租户审计日志数据访问层"""
    
    def __init__(self, use_memory_storage: bool = False):
        """初始化审计日志仓库
        
        Args:
            use_memory_storage: 是否使用内存存储
        """
        self._use_memory_storage = use_memory_storage
        self._max_memory_logs = 10000  # 内存中最大日志数量
        
        if use_memory_storage:
            self._logs: Dict[str, List[Dict]] = {}
            self._db_manager = None
        else:
            try:
                from backend.modules.database.manager import get_database_manager
                self._db_manager = get_database_manager()
            except ImportError:
                logger.warning("无法导入数据库服务，回退到内存存储模式")
                self._use_memory_storage = True
                self._logs: Dict[str, List[Dict]] = {}
                self._db_manager = None
    
    def create(self, log_data: Dict[str, Any]) -> TenantAuditLog:
        """创建审计日志
        
        Args:
            log_data: 日志数据
            
        Returns:
            创建的日志
        """
        try:
            tenant_id = log_data['tenant_id']
            
            if self._use_memory_storage:
                if tenant_id not in self._logs:
                    self._logs[tenant_id] = []
                
                self._logs[tenant_id].append(log_data)
                
                # 限制内存中的日志数量
                if len(self._logs[tenant_id]) > self._max_memory_logs:
                    self._logs[tenant_id] = self._logs[tenant_id][-self._max_memory_logs // 2:]
                
                return log_data
            
            with self._db_manager.get_db_session() as db:
                # 处理 details 字段
                details = log_data.get('details', {})
                if isinstance(details, dict):
                    details = json.dumps(details)
                
                audit_log = TenantAuditLog(
                    id=log_data.get('id'),
                    tenant_id=tenant_id,
                    user_id=log_data['user_id'],
                    action=log_data['action'],
                    resource_type=log_data['resource_type'],
                    resource_id=log_data.get('resource_id'),
                    details=details,
                    ip_address=log_data.get('ip_address'),
                    user_agent=log_data.get('user_agent'),
                    timestamp=log_data.get('timestamp', datetime.utcnow())
                )
                db.add(audit_log)
                db.commit()
                db.refresh(audit_log)
                return audit_log
                
        except Exception as e:
            logger.error(f"创建审计日志失败: {e}")
            # 审计日志创建失败不应该影响主业务
            return log_data
    
    def get_by_id(self, log_id: str) -> Optional[TenantAuditLog]:
        """根据ID获取审计日志
        
        Args:
            log_id: 日志ID
            
        Returns:
            日志对象
        """
        try:
            if self._use_memory_storage:
                for logs in self._logs.values():
                    for log in logs:
                        if log.get('id') == log_id:
                            return log
                return None
            
            with self._db_manager.get_db_session() as db:
                return db.query(TenantAuditLog).filter(
                    TenantAuditLog.id == log_id
                ).first()
                
        except Exception as e:
            logger.error(f"获取审计日志失败: {e}")
            return None
    
    def list_by_tenant(self, tenant_id: str, 
                      action: Optional[str] = None,
                      resource_type: Optional[str] = None,
                      user_id: Optional[str] = None,
                      start_date: Optional[datetime] = None,
                      end_date: Optional[datetime] = None,
                      limit: int = 100, offset: int = 0) -> Tuple[List[TenantAuditLog], int]:
        """获取租户的审计日志列表
        
        Args:
            tenant_id: 租户ID
            action: 操作类型过滤
            resource_type: 资源类型过滤
            user_id: 用户ID过滤
            start_date: 开始日期
            end_date: 结束日期
            limit: 限制数量
            offset: 偏移量
            
        Returns:
            (日志列表, 总数)
        """
        try:
            if self._use_memory_storage:
                logs = self._logs.get(tenant_id, [])
                
                # 应用过滤器
                if action:
                    logs = [l for l in logs if l.get('action') == action]
                if resource_type:
                    logs = [l for l in logs if l.get('resource_type') == resource_type]
                if user_id:
                    logs = [l for l in logs if l.get('user_id') == user_id]
                if start_date:
                    logs = [l for l in logs if l.get('timestamp', datetime.min) >= start_date]
                if end_date:
                    logs = [l for l in logs if l.get('timestamp', datetime.max) <= end_date]
                
                # 按时间倒序排序
                logs.sort(key=lambda x: x.get('timestamp', datetime.min), reverse=True)
                
                total = len(logs)
                return logs[offset:offset + limit], total
            
            with self._db_manager.get_db_session() as db:
                query = db.query(TenantAuditLog).filter(
                    TenantAuditLog.tenant_id == tenant_id
                )
                
                if action:
                    query = query.filter(TenantAuditLog.action == action)
                if resource_type:
                    query = query.filter(TenantAuditLog.resource_type == resource_type)
                if user_id:
                    query = query.filter(TenantAuditLog.user_id == user_id)
                if start_date:
                    query = query.filter(TenantAuditLog.timestamp >= start_date)
                if end_date:
                    query = query.filter(TenantAuditLog.timestamp <= end_date)
                
                total = query.count()
                logs = query.order_by(
                    TenantAuditLog.timestamp.desc()
                ).offset(offset).limit(limit).all()
                
                return logs, total
                
        except Exception as e:
            logger.error(f"获取审计日志列表失败: {e}")
            return [], 0
    
    def count_by_tenant(self, tenant_id: str, 
                       action: Optional[str] = None,
                       since: Optional[datetime] = None) -> int:
        """获取租户审计日志数量
        
        Args:
            tenant_id: 租户ID
            action: 操作类型过滤
            since: 起始时间
            
        Returns:
            日志数量
        """
        try:
            if self._use_memory_storage:
                logs = self._logs.get(tenant_id, [])
                if action:
                    logs = [l for l in logs if l.get('action') == action]
                if since:
                    logs = [l for l in logs if l.get('timestamp', datetime.min) >= since]
                return len(logs)
            
            with self._db_manager.get_db_session() as db:
                query = db.query(TenantAuditLog).filter(
                    TenantAuditLog.tenant_id == tenant_id
                )
                if action:
                    query = query.filter(TenantAuditLog.action == action)
                if since:
                    query = query.filter(TenantAuditLog.timestamp >= since)
                return query.count()
                
        except Exception as e:
            logger.error(f"获取审计日志数量失败: {e}")
            return 0
    
    def delete_before(self, tenant_id: str, before_date: datetime) -> int:
        """删除指定日期之前的审计日志
        
        Args:
            tenant_id: 租户ID
            before_date: 日期阈值
            
        Returns:
            删除的数量
        """
        try:
            if self._use_memory_storage:
                if tenant_id in self._logs:
                    original_count = len(self._logs[tenant_id])
                    self._logs[tenant_id] = [
                        l for l in self._logs[tenant_id]
                        if l.get('timestamp', datetime.max) >= before_date
                    ]
                    return original_count - len(self._logs[tenant_id])
                return 0
            
            with self._db_manager.get_db_session() as db:
                count = db.query(TenantAuditLog).filter(
                    TenantAuditLog.tenant_id == tenant_id,
                    TenantAuditLog.timestamp < before_date
                ).delete()
                db.commit()
                return count
                
        except Exception as e:
            logger.error(f"删除审计日志失败: {e}")
            return 0
    
    def delete_all_by_tenant(self, tenant_id: str) -> int:
        """删除租户的所有审计日志
        
        Args:
            tenant_id: 租户ID
            
        Returns:
            删除的数量
        """
        try:
            if self._use_memory_storage:
                count = len(self._logs.get(tenant_id, []))
                self._logs.pop(tenant_id, None)
                return count
            
            with self._db_manager.get_db_session() as db:
                count = db.query(TenantAuditLog).filter(
                    TenantAuditLog.tenant_id == tenant_id
                ).delete()
                db.commit()
                return count
                
        except Exception as e:
            logger.error(f"删除租户审计日志失败: {e}")
            return 0
    
    def get_action_summary(self, tenant_id: str, 
                          since: Optional[datetime] = None) -> Dict[str, int]:
        """获取操作类型汇总
        
        Args:
            tenant_id: 租户ID
            since: 起始时间
            
        Returns:
            {action: count}
        """
        try:
            if self._use_memory_storage:
                logs = self._logs.get(tenant_id, [])
                if since:
                    logs = [l for l in logs if l.get('timestamp', datetime.min) >= since]
                
                summary = {}
                for log in logs:
                    action = log.get('action', 'unknown')
                    summary[action] = summary.get(action, 0) + 1
                return summary
            
            with self._db_manager.get_db_session() as db:
                from sqlalchemy import func
                
                query = db.query(
                    TenantAuditLog.action,
                    func.count(TenantAuditLog.id)
                ).filter(
                    TenantAuditLog.tenant_id == tenant_id
                )
                
                if since:
                    query = query.filter(TenantAuditLog.timestamp >= since)
                
                results = query.group_by(TenantAuditLog.action).all()
                return {action: count for action, count in results}
                
        except Exception as e:
            logger.error(f"获取操作汇总失败: {e}")
            return {}


# 全局实例
_audit_log_repository: Optional[TenantAuditLogRepository] = None


def get_tenant_audit_log_repository(use_memory: bool = False) -> TenantAuditLogRepository:
    """获取审计日志仓库实例"""
    global _audit_log_repository
    if _audit_log_repository is None:
        _audit_log_repository = TenantAuditLogRepository(use_memory_storage=use_memory)
    return _audit_log_repository



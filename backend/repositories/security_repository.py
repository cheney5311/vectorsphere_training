# -*- coding: utf-8 -*-
"""安全仓库层

提供安全相关数据的持久化访问接口，包括：
- 用户会话管理
- 审计日志管理
- 角色权限管理
- 加密密钥管理
- 合规记录管理
"""

import logging
import uuid
from typing import Optional, List, Dict, Any, Union
from datetime import datetime, timedelta
from sqlalchemy import func, and_, or_, desc, asc

logger = logging.getLogger(__name__)


class UserSessionRepository:
    """用户会话仓库"""
    
    def __init__(self, db_service=None, use_memory_storage: bool = False):
        self._use_memory_storage = use_memory_storage
        self._db_service = None
        self._sessions: Dict[str, Dict] = {}
        
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
        except ImportError as e:
            logger.warning(f"Database manager not available: {e}, using memory storage")
            self._use_memory_storage = True
    
    def create(self, session_data: Dict[str, Any]) -> Optional[str]:
        """创建会话"""
        session_id = str(uuid.uuid4())
        
        if self._use_memory_storage:
            session_data['id'] = session_id
            session_data['created_at'] = datetime.utcnow()
            self._sessions[session_id] = session_data
            return session_id
        
        try:
            from backend.schemas.security_models import UserSession
            
            with self._db_service.get_session() as db:
                session = UserSession(
                    id=session_id,
                    session_token=session_data.get('session_token'),
                    user_id=session_data.get('user_id'),
                    tenant_id=session_data.get('tenant_id'),
                    auth_method=session_data.get('auth_method', 'password'),
                    mfa_verified=session_data.get('mfa_verified', False),
                    status=session_data.get('status', 'active'),
                    ip_address=session_data.get('ip_address'),
                    user_agent=session_data.get('user_agent'),
                    device_fingerprint=session_data.get('device_fingerprint'),
                    device_type=session_data.get('device_type'),
                    risk_score=session_data.get('risk_score', 0.0),
                    is_trusted_device=session_data.get('is_trusted_device', False),
                    expires_at=session_data.get('expires_at'),
                    last_activity_at=datetime.utcnow(),
                    metadata=session_data.get('metadata')
                )
                db.add(session)
                db.commit()
                return session_id
                
        except Exception as e:
            logger.error(f"Failed to create session: {e}")
            return None
    
    def get_by_token(self, session_token: str, tenant_id: str = None) -> Optional[Dict]:
        """根据令牌获取会话"""
        if self._use_memory_storage:
            for session in self._sessions.values():
                if session.get('session_token') == session_token:
                    if tenant_id and session.get('tenant_id') != tenant_id:
                        continue
                    return session
            return None
        
        try:
            from backend.schemas.security_models import UserSession
            
            with self._db_service.get_session() as db:
                query = db.query(UserSession).filter(
                    UserSession.session_token == session_token
                )
                if tenant_id:
                    query = query.filter(UserSession.tenant_id == tenant_id)
                
                session = query.first()
                return session.to_dict() if session else None
                
        except Exception as e:
            logger.error(f"Failed to get session by token: {e}")
            return None
    
    def get_by_user(self, user_id: str, tenant_id: str = None, 
                   status: str = None) -> List[Dict]:
        """获取用户的所有会话"""
        if self._use_memory_storage:
            sessions = []
            for session in self._sessions.values():
                if session.get('user_id') != user_id:
                    continue
                if tenant_id and session.get('tenant_id') != tenant_id:
                    continue
                if status and session.get('status') != status:
                    continue
                sessions.append(session)
            return sessions
        
        try:
            from backend.schemas.security_models import UserSession
            
            with self._db_service.get_session() as db:
                query = db.query(UserSession).filter(
                    UserSession.user_id == user_id
                )
                if tenant_id:
                    query = query.filter(UserSession.tenant_id == tenant_id)
                if status:
                    query = query.filter(UserSession.status == status)
                
                sessions = query.order_by(desc(UserSession.created_at)).all()
                return [s.to_dict() for s in sessions]
                
        except Exception as e:
            logger.error(f"Failed to get sessions by user: {e}")
            return []
    
    def update_status(self, session_id: str, status: str) -> bool:
        """更新会话状态"""
        if self._use_memory_storage:
            if session_id in self._sessions:
                self._sessions[session_id]['status'] = status
                self._sessions[session_id]['updated_at'] = datetime.utcnow()
                return True
            return False
        
        try:
            from backend.schemas.security_models import UserSession
            
            with self._db_service.get_session() as db:
                session = db.query(UserSession).filter(
                    UserSession.id == session_id
                ).first()
                
                if session:
                    session.status = status
                    session.updated_at = datetime.utcnow()
                    db.commit()
                    return True
                return False
                
        except Exception as e:
            logger.error(f"Failed to update session status: {e}")
            return False
    
    def update_activity(self, session_id: str) -> bool:
        """更新最后活动时间"""
        if self._use_memory_storage:
            if session_id in self._sessions:
                self._sessions[session_id]['last_activity_at'] = datetime.utcnow()
                return True
            return False
        
        try:
            from backend.schemas.security_models import UserSession
            
            with self._db_service.get_session() as db:
                session = db.query(UserSession).filter(
                    UserSession.id == session_id
                ).first()
                
                if session:
                    session.last_activity_at = datetime.utcnow()
                    db.commit()
                    return True
                return False
                
        except Exception as e:
            logger.error(f"Failed to update session activity: {e}")
            return False
    
    def delete_expired(self, tenant_id: str = None) -> int:
        """删除过期会话"""
        now = datetime.utcnow()
        deleted_count = 0
        
        if self._use_memory_storage:
            to_delete = []
            for sid, session in self._sessions.items():
                if tenant_id and session.get('tenant_id') != tenant_id:
                    continue
                expires_at = session.get('expires_at')
                if expires_at and expires_at < now:
                    to_delete.append(sid)
            
            for sid in to_delete:
                del self._sessions[sid]
                deleted_count += 1
            return deleted_count
        
        try:
            from backend.schemas.security_models import UserSession
            
            with self._db_service.get_session() as db:
                query = db.query(UserSession).filter(
                    UserSession.expires_at < now
                )
                if tenant_id:
                    query = query.filter(UserSession.tenant_id == tenant_id)
                
                deleted_count = query.delete()
                db.commit()
                return deleted_count
                
        except Exception as e:
            logger.error(f"Failed to delete expired sessions: {e}")
            return 0
    
    def count_active(self, tenant_id: str = None, user_id: str = None) -> int:
        """统计活跃会话数"""
        if self._use_memory_storage:
            count = 0
            for session in self._sessions.values():
                if session.get('status') != 'active':
                    continue
                if tenant_id and session.get('tenant_id') != tenant_id:
                    continue
                if user_id and session.get('user_id') != user_id:
                    continue
                count += 1
            return count
        
        try:
            from backend.schemas.security_models import UserSession
            
            with self._db_service.get_session() as db:
                query = db.query(func.count(UserSession.id)).filter(
                    UserSession.status == 'active'
                )
                if tenant_id:
                    query = query.filter(UserSession.tenant_id == tenant_id)
                if user_id:
                    query = query.filter(UserSession.user_id == user_id)
                
                return query.scalar() or 0
                
        except Exception as e:
            logger.error(f"Failed to count active sessions: {e}")
            return 0


class SecurityAuditLogRepository:
    """安全审计日志仓库"""
    
    def __init__(self, db_service=None, use_memory_storage: bool = False):
        self._use_memory_storage = use_memory_storage
        self._db_service = None
        self._logs: List[Dict] = []
        
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
        except ImportError as e:
            logger.warning(f"Database manager not available: {e}, using memory storage")
            self._use_memory_storage = True
    
    def create(self, log_data: Dict[str, Any]) -> Optional[str]:
        """创建审计日志"""
        log_id = str(uuid.uuid4())
        
        if self._use_memory_storage:
            log_data['id'] = log_id
            log_data['created_at'] = datetime.utcnow()
            self._logs.append(log_data)
            return log_id
        
        try:
            from backend.schemas.security_models import SecurityAuditLog
            
            with self._db_service.get_session() as db:
                log = SecurityAuditLog(
                    id=log_id,
                    tenant_id=log_data.get('tenant_id'),
                    event_type=log_data.get('event_type'),
                    event_level=log_data.get('event_level', 'info'),
                    user_id=log_data.get('user_id'),
                    session_id=log_data.get('session_id'),
                    source_ip=log_data.get('source_ip'),
                    user_agent=log_data.get('user_agent'),
                    resource=log_data.get('resource'),
                    action=log_data.get('action'),
                    result=log_data.get('result', 'success'),
                    message=log_data.get('message'),
                    details=log_data.get('details'),
                    risk_score=log_data.get('risk_score', 0.0),
                    tags=log_data.get('tags')
                )
                db.add(log)
                db.commit()
                return log_id
                
        except Exception as e:
            logger.error(f"Failed to create audit log: {e}")
            return None
    
    def query(
        self,
        tenant_id: str,
        start_time: datetime = None,
        end_time: datetime = None,
        event_types: List[str] = None,
        event_levels: List[str] = None,
        user_ids: List[str] = None,
        resources: List[str] = None,
        results: List[str] = None,
        min_risk_score: float = None,
        max_risk_score: float = None,
        tags: List[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict]:
        """查询审计日志"""
        if self._use_memory_storage:
            filtered = []
            for log in self._logs:
                if log.get('tenant_id') != tenant_id:
                    continue
                if start_time and log.get('created_at', datetime.min) < start_time:
                    continue
                if end_time and log.get('created_at', datetime.max) > end_time:
                    continue
                if event_types and log.get('event_type') not in event_types:
                    continue
                if event_levels and log.get('event_level') not in event_levels:
                    continue
                if user_ids and log.get('user_id') not in user_ids:
                    continue
                if resources and log.get('resource') not in resources:
                    continue
                if results and log.get('result') not in results:
                    continue
                if min_risk_score and log.get('risk_score', 0) < min_risk_score:
                    continue
                if max_risk_score and log.get('risk_score', 0) > max_risk_score:
                    continue
                filtered.append(log)
            
            # 排序并分页
            filtered.sort(key=lambda x: x.get('created_at', datetime.min), reverse=True)
            return filtered[offset:offset + limit]
        
        try:
            from backend.schemas.security_models import SecurityAuditLog
            
            with self._db_service.get_session() as db:
                query = db.query(SecurityAuditLog).filter(
                    SecurityAuditLog.tenant_id == tenant_id
                )
                
                if start_time:
                    query = query.filter(SecurityAuditLog.created_at >= start_time)
                if end_time:
                    query = query.filter(SecurityAuditLog.created_at <= end_time)
                if event_types:
                    query = query.filter(SecurityAuditLog.event_type.in_(event_types))
                if event_levels:
                    query = query.filter(SecurityAuditLog.event_level.in_(event_levels))
                if user_ids:
                    query = query.filter(SecurityAuditLog.user_id.in_(user_ids))
                if resources:
                    query = query.filter(SecurityAuditLog.resource.in_(resources))
                if results:
                    query = query.filter(SecurityAuditLog.result.in_(results))
                if min_risk_score is not None:
                    query = query.filter(SecurityAuditLog.risk_score >= min_risk_score)
                if max_risk_score is not None:
                    query = query.filter(SecurityAuditLog.risk_score <= max_risk_score)
                
                logs = query.order_by(desc(SecurityAuditLog.created_at)).offset(offset).limit(limit).all()
                return [log.to_dict() for log in logs]
                
        except Exception as e:
            logger.error(f"Failed to query audit logs: {e}")
            return []
    
    def get_statistics(
        self,
        tenant_id: str,
        start_time: datetime = None,
        end_time: datetime = None
    ) -> Dict[str, Any]:
        """获取审计日志统计"""
        if self._use_memory_storage:
            stats = {
                'total': 0,
                'by_event_type': {},
                'by_level': {},
                'by_result': {},
                'high_risk_count': 0
            }
            
            for log in self._logs:
                if log.get('tenant_id') != tenant_id:
                    continue
                if start_time and log.get('created_at', datetime.min) < start_time:
                    continue
                if end_time and log.get('created_at', datetime.max) > end_time:
                    continue
                
                stats['total'] += 1
                
                event_type = log.get('event_type', 'unknown')
                stats['by_event_type'][event_type] = stats['by_event_type'].get(event_type, 0) + 1
                
                level = log.get('event_level', 'info')
                stats['by_level'][level] = stats['by_level'].get(level, 0) + 1
                
                result = log.get('result', 'unknown')
                stats['by_result'][result] = stats['by_result'].get(result, 0) + 1
                
                if log.get('risk_score', 0) > 0.7:
                    stats['high_risk_count'] += 1
            
            return stats
        
        try:
            from backend.schemas.security_models import SecurityAuditLog
            
            with self._db_service.get_session() as db:
                base_filters = [SecurityAuditLog.tenant_id == tenant_id]
                if start_time:
                    base_filters.append(SecurityAuditLog.created_at >= start_time)
                if end_time:
                    base_filters.append(SecurityAuditLog.created_at <= end_time)
                
                # 总数
                total = db.query(func.count(SecurityAuditLog.id)).filter(
                    and_(*base_filters)
                ).scalar() or 0
                
                # 按事件类型统计
                by_event_type = {}
                event_type_results = db.query(
                    SecurityAuditLog.event_type,
                    func.count(SecurityAuditLog.id)
                ).filter(and_(*base_filters)).group_by(
                    SecurityAuditLog.event_type
                ).all()
                for event_type, count in event_type_results:
                    by_event_type[event_type] = count
                
                # 按级别统计
                by_level = {}
                level_results = db.query(
                    SecurityAuditLog.event_level,
                    func.count(SecurityAuditLog.id)
                ).filter(and_(*base_filters)).group_by(
                    SecurityAuditLog.event_level
                ).all()
                for level, count in level_results:
                    by_level[level] = count
                
                # 高风险事件数
                high_risk_count = db.query(func.count(SecurityAuditLog.id)).filter(
                    and_(*base_filters, SecurityAuditLog.risk_score > 0.7)
                ).scalar() or 0
                
                return {
                    'total': total,
                    'by_event_type': by_event_type,
                    'by_level': by_level,
                    'high_risk_count': high_risk_count
                }
                
        except Exception as e:
            logger.error(f"Failed to get audit statistics: {e}")
            return {'total': 0, 'by_event_type': {}, 'by_level': {}, 'high_risk_count': 0}
    
    def cleanup_old_logs(self, tenant_id: str, retention_days: int) -> int:
        """清理旧日志"""
        cutoff_date = datetime.utcnow() - timedelta(days=retention_days)
        
        if self._use_memory_storage:
            initial_count = len(self._logs)
            self._logs = [
                log for log in self._logs
                if log.get('tenant_id') != tenant_id or log.get('created_at', datetime.max) >= cutoff_date
            ]
            return initial_count - len(self._logs)
        
        try:
            from backend.schemas.security_models import SecurityAuditLog
            
            with self._db_service.get_session() as db:
                deleted = db.query(SecurityAuditLog).filter(
                    SecurityAuditLog.tenant_id == tenant_id,
                    SecurityAuditLog.created_at < cutoff_date
                ).delete()
                db.commit()
                return deleted
                
        except Exception as e:
            logger.error(f"Failed to cleanup old logs: {e}")
            return 0


class UserRoleRepository:
    """用户角色仓库"""
    
    def __init__(self, db_service=None, use_memory_storage: bool = False):
        self._use_memory_storage = use_memory_storage
        self._db_service = None
        self._roles: Dict[str, List[Dict]] = {}  # user_id -> roles
        
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
        except ImportError as e:
            logger.warning(f"Database manager not available: {e}, using memory storage")
            self._use_memory_storage = True
    
    def assign_role(
        self,
        user_id: str,
        role: str,
        tenant_id: str,
        assigned_by: str = None,
        expires_at: datetime = None
    ) -> Optional[str]:
        """分配角色"""
        role_id = str(uuid.uuid4())
        
        if self._use_memory_storage:
            if user_id not in self._roles:
                self._roles[user_id] = []
            
            # 检查是否已存在
            for r in self._roles[user_id]:
                if r.get('role') == role and r.get('tenant_id') == tenant_id:
                    return r.get('id')  # 已存在
            
            role_data = {
                'id': role_id,
                'user_id': user_id,
                'role': role,
                'tenant_id': tenant_id,
                'assigned_by': assigned_by,
                'assigned_at': datetime.utcnow(),
                'expires_at': expires_at,
                'is_active': True,
                'created_at': datetime.utcnow()
            }
            self._roles[user_id].append(role_data)
            return role_id
        
        try:
            from backend.schemas.security_models import UserRole
            
            with self._db_service.get_session() as db:
                # 检查是否已存在
                existing = db.query(UserRole).filter(
                    UserRole.user_id == user_id,
                    UserRole.role == role,
                    UserRole.tenant_id == tenant_id,
                    UserRole.is_active == True
                ).first()
                
                if existing:
                    return str(existing.id)
                
                user_role = UserRole(
                    id=role_id,
                    user_id=user_id,
                    role=role,
                    tenant_id=tenant_id,
                    assigned_by=assigned_by,
                    assigned_at=datetime.utcnow(),
                    expires_at=expires_at,
                    is_active=True
                )
                db.add(user_role)
                db.commit()
                return role_id
                
        except Exception as e:
            logger.error(f"Failed to assign role: {e}")
            return None
    
    def revoke_role(self, user_id: str, role: str, tenant_id: str) -> bool:
        """撤销角色"""
        if self._use_memory_storage:
            if user_id in self._roles:
                for r in self._roles[user_id]:
                    if r.get('role') == role and r.get('tenant_id') == tenant_id:
                        r['is_active'] = False
                        r['updated_at'] = datetime.utcnow()
                        return True
            return False
        
        try:
            from backend.schemas.security_models import UserRole
            
            with self._db_service.get_session() as db:
                user_role = db.query(UserRole).filter(
                    UserRole.user_id == user_id,
                    UserRole.role == role,
                    UserRole.tenant_id == tenant_id,
                    UserRole.is_active == True
                ).first()
                
                if user_role:
                    user_role.is_active = False
                    user_role.updated_at = datetime.utcnow()
                    db.commit()
                    return True
                return False
                
        except Exception as e:
            logger.error(f"Failed to revoke role: {e}")
            return False
    
    def get_user_roles(self, user_id: str, tenant_id: str = None) -> List[str]:
        """获取用户角色列表"""
        if self._use_memory_storage:
            roles = []
            for r in self._roles.get(user_id, []):
                if not r.get('is_active'):
                    continue
                if tenant_id and r.get('tenant_id') != tenant_id:
                    continue
                # 检查过期
                expires_at = r.get('expires_at')
                if expires_at and expires_at < datetime.utcnow():
                    continue
                roles.append(r.get('role'))
            return roles
        
        try:
            from backend.schemas.security_models import UserRole
            
            with self._db_service.get_session() as db:
                query = db.query(UserRole.role).filter(
                    UserRole.user_id == user_id,
                    UserRole.is_active == True
                )
                if tenant_id:
                    query = query.filter(UserRole.tenant_id == tenant_id)
                
                # 过滤过期的
                query = query.filter(
                    or_(
                        UserRole.expires_at.is_(None),
                        UserRole.expires_at > datetime.utcnow()
                    )
                )
                
                results = query.all()
                return [r[0] for r in results]
                
        except Exception as e:
            logger.error(f"Failed to get user roles: {e}")
            return []
    
    def check_role(self, user_id: str, role: str, tenant_id: str) -> bool:
        """检查用户是否拥有角色"""
        return role in self.get_user_roles(user_id, tenant_id)


class AccessPolicyRepository:
    """访问策略仓库"""
    
    def __init__(self, db_service=None, use_memory_storage: bool = False):
        self._use_memory_storage = use_memory_storage
        self._db_service = None
        self._policies: Dict[str, Dict] = {}
        
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
        except ImportError as e:
            logger.warning(f"Database manager not available: {e}, using memory storage")
            self._use_memory_storage = True
    
    def create(self, policy_data: Dict[str, Any]) -> Optional[str]:
        """创建访问策略"""
        policy_id = policy_data.get('policy_id') or str(uuid.uuid4())
        
        if self._use_memory_storage:
            policy_data['policy_id'] = policy_id
            policy_data['created_at'] = datetime.utcnow()
            self._policies[policy_id] = policy_data
            return policy_id
        
        try:
            from backend.schemas.security_models import AccessPolicy
            
            with self._db_service.get_session() as db:
                policy = AccessPolicy(
                    policy_id=policy_id,
                    name=policy_data.get('name'),
                    description=policy_data.get('description'),
                    tenant_id=policy_data.get('tenant_id'),
                    effect=policy_data.get('effect', 'allow'),
                    principals=policy_data.get('principals', []),
                    resources=policy_data.get('resources', []),
                    actions=policy_data.get('actions', []),
                    conditions=policy_data.get('conditions', {}),
                    priority=policy_data.get('priority', 100),
                    is_active=policy_data.get('is_active', True)
                )
                db.add(policy)
                db.commit()
                return policy_id
                
        except Exception as e:
            logger.error(f"Failed to create policy: {e}")
            return None
    
    def get_by_id(self, policy_id: str, tenant_id: str = None) -> Optional[Dict]:
        """根据ID获取策略"""
        if self._use_memory_storage:
            policy = self._policies.get(policy_id)
            if policy and (not tenant_id or policy.get('tenant_id') == tenant_id):
                return policy
            return None
        
        try:
            from backend.schemas.security_models import AccessPolicy
            
            with self._db_service.get_session() as db:
                query = db.query(AccessPolicy).filter(
                    AccessPolicy.policy_id == policy_id
                )
                if tenant_id:
                    query = query.filter(AccessPolicy.tenant_id == tenant_id)
                
                policy = query.first()
                return policy.to_dict() if policy else None
                
        except Exception as e:
            logger.error(f"Failed to get policy: {e}")
            return None
    
    def list_by_tenant(self, tenant_id: str, is_active: bool = True) -> List[Dict]:
        """列出租户的所有策略"""
        if self._use_memory_storage:
            policies = []
            for policy in self._policies.values():
                if policy.get('tenant_id') != tenant_id:
                    continue
                if is_active is not None and policy.get('is_active') != is_active:
                    continue
                policies.append(policy)
            # 按优先级排序
            policies.sort(key=lambda x: x.get('priority', 100))
            return policies
        
        try:
            from backend.schemas.security_models import AccessPolicy
            
            with self._db_service.get_session() as db:
                query = db.query(AccessPolicy).filter(
                    AccessPolicy.tenant_id == tenant_id
                )
                if is_active is not None:
                    query = query.filter(AccessPolicy.is_active == is_active)
                
                policies = query.order_by(AccessPolicy.priority).all()
                return [p.to_dict() for p in policies]
                
        except Exception as e:
            logger.error(f"Failed to list policies: {e}")
            return []
    
    def update(self, policy_id: str, update_data: Dict[str, Any], tenant_id: str = None) -> bool:
        """更新策略"""
        if self._use_memory_storage:
            if policy_id in self._policies:
                policy = self._policies[policy_id]
                if tenant_id and policy.get('tenant_id') != tenant_id:
                    return False
                policy.update(update_data)
                policy['updated_at'] = datetime.utcnow()
                return True
            return False
        
        try:
            from backend.schemas.security_models import AccessPolicy
            
            with self._db_service.get_session() as db:
                query = db.query(AccessPolicy).filter(
                    AccessPolicy.policy_id == policy_id
                )
                if tenant_id:
                    query = query.filter(AccessPolicy.tenant_id == tenant_id)
                
                policy = query.first()
                if policy:
                    for key, value in update_data.items():
                        if hasattr(policy, key):
                            setattr(policy, key, value)
                    policy.updated_at = datetime.utcnow()
                    db.commit()
                    return True
                return False
                
        except Exception as e:
            logger.error(f"Failed to update policy: {e}")
            return False
    
    def delete(self, policy_id: str, tenant_id: str = None) -> bool:
        """删除策略"""
        if self._use_memory_storage:
            if policy_id in self._policies:
                policy = self._policies[policy_id]
                if tenant_id and policy.get('tenant_id') != tenant_id:
                    return False
                del self._policies[policy_id]
                return True
            return False
        
        try:
            from backend.schemas.security_models import AccessPolicy
            
            with self._db_service.get_session() as db:
                query = db.query(AccessPolicy).filter(
                    AccessPolicy.policy_id == policy_id
                )
                if tenant_id:
                    query = query.filter(AccessPolicy.tenant_id == tenant_id)
                
                deleted = query.delete()
                db.commit()
                return deleted > 0
                
        except Exception as e:
            logger.error(f"Failed to delete policy: {e}")
            return False


class EncryptionKeyRepository:
    """加密密钥仓库"""
    
    def __init__(self, db_service=None, use_memory_storage: bool = False):
        self._use_memory_storage = use_memory_storage
        self._db_service = None
        self._keys: Dict[str, Dict] = {}
        
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
        except ImportError as e:
            logger.warning(f"Database manager not available: {e}, using memory storage")
            self._use_memory_storage = True
    
    def create(self, key_data: Dict[str, Any]) -> Optional[str]:
        """创建密钥记录"""
        key_id = key_data.get('key_id') or str(uuid.uuid4())
        
        if self._use_memory_storage:
            key_data['key_id'] = key_id
            key_data['created_at'] = datetime.utcnow()
            self._keys[key_id] = key_data
            return key_id
        
        try:
            from backend.schemas.security_models import EncryptionKey
            
            with self._db_service.get_session() as db:
                key = EncryptionKey(
                    key_id=key_id,
                    name=key_data.get('name'),
                    description=key_data.get('description'),
                    tenant_id=key_data.get('tenant_id'),
                    key_type=key_data.get('key_type'),
                    algorithm=key_data.get('algorithm'),
                    key_size=key_data.get('key_size'),
                    encrypted_key_data=key_data.get('encrypted_key_data'),
                    key_checksum=key_data.get('key_checksum'),
                    status=key_data.get('status', 'active'),
                    expires_at=key_data.get('expires_at'),
                    metadata=key_data.get('metadata')
                )
                db.add(key)
                db.commit()
                return key_id
                
        except Exception as e:
            logger.error(f"Failed to create encryption key: {e}")
            return None
    
    def get_by_id(self, key_id: str, tenant_id: str = None) -> Optional[Dict]:
        """根据ID获取密钥"""
        if self._use_memory_storage:
            key = self._keys.get(key_id)
            if key and (not tenant_id or key.get('tenant_id') == tenant_id):
                return key
            return None
        
        try:
            from backend.schemas.security_models import EncryptionKey
            
            with self._db_service.get_session() as db:
                query = db.query(EncryptionKey).filter(
                    EncryptionKey.key_id == key_id
                )
                if tenant_id:
                    query = query.filter(EncryptionKey.tenant_id == tenant_id)
                
                key = query.first()
                return key.to_dict() if key else None
                
        except Exception as e:
            logger.error(f"Failed to get encryption key: {e}")
            return None
    
    def list_by_tenant(
        self,
        tenant_id: str,
        status: str = None,
        algorithm: str = None
    ) -> List[Dict]:
        """列出租户的密钥"""
        if self._use_memory_storage:
            keys = []
            for key in self._keys.values():
                if key.get('tenant_id') != tenant_id:
                    continue
                if status and key.get('status') != status:
                    continue
                if algorithm and key.get('algorithm') != algorithm:
                    continue
                keys.append(key)
            return keys
        
        try:
            from backend.schemas.security_models import EncryptionKey
            
            with self._db_service.get_session() as db:
                query = db.query(EncryptionKey).filter(
                    EncryptionKey.tenant_id == tenant_id
                )
                if status:
                    query = query.filter(EncryptionKey.status == status)
                if algorithm:
                    query = query.filter(EncryptionKey.algorithm == algorithm)
                
                keys = query.order_by(desc(EncryptionKey.created_at)).all()
                return [k.to_dict() for k in keys]
                
        except Exception as e:
            logger.error(f"Failed to list encryption keys: {e}")
            return []
    
    def update_status(self, key_id: str, status: str, tenant_id: str = None) -> bool:
        """更新密钥状态"""
        if self._use_memory_storage:
            if key_id in self._keys:
                key = self._keys[key_id]
                if tenant_id and key.get('tenant_id') != tenant_id:
                    return False
                key['status'] = status
                key['updated_at'] = datetime.utcnow()
                return True
            return False
        
        try:
            from backend.schemas.security_models import EncryptionKey
            
            with self._db_service.get_session() as db:
                query = db.query(EncryptionKey).filter(
                    EncryptionKey.key_id == key_id
                )
                if tenant_id:
                    query = query.filter(EncryptionKey.tenant_id == tenant_id)
                
                key = query.first()
                if key:
                    key.status = status
                    key.updated_at = datetime.utcnow()
                    db.commit()
                    return True
                return False
                
        except Exception as e:
            logger.error(f"Failed to update key status: {e}")
            return False
    
    def increment_usage(self, key_id: str) -> bool:
        """增加使用次数"""
        if self._use_memory_storage:
            if key_id in self._keys:
                self._keys[key_id]['usage_count'] = self._keys[key_id].get('usage_count', 0) + 1
                self._keys[key_id]['last_used_at'] = datetime.utcnow()
                return True
            return False
        
        try:
            from backend.schemas.security_models import EncryptionKey
            
            with self._db_service.get_session() as db:
                key = db.query(EncryptionKey).filter(
                    EncryptionKey.key_id == key_id
                ).first()
                
                if key:
                    key.usage_count = (key.usage_count or 0) + 1
                    key.last_used_at = datetime.utcnow()
                    db.commit()
                    return True
                return False
                
        except Exception as e:
            logger.error(f"Failed to increment key usage: {e}")
            return False


class DataProcessingRecordRepository:
    """数据处理记录仓库"""
    
    def __init__(self, db_service=None, use_memory_storage: bool = False):
        self._use_memory_storage = use_memory_storage
        self._db_service = None
        self._records: Dict[str, Dict] = {}
        
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
        except ImportError as e:
            logger.warning(f"Database manager not available: {e}, using memory storage")
            self._use_memory_storage = True
    
    def create(self, record_data: Dict[str, Any]) -> Optional[str]:
        """创建数据处理记录"""
        record_id = record_data.get('record_id') or str(uuid.uuid4())
        
        if self._use_memory_storage:
            record_data['record_id'] = record_id
            record_data['created_at'] = datetime.utcnow()
            self._records[record_id] = record_data
            return record_id
        
        try:
            from backend.schemas.security_models import DataProcessingRecord
            
            with self._db_service.get_session() as db:
                record = DataProcessingRecord(
                    record_id=record_id,
                    tenant_id=record_data.get('tenant_id'),
                    data_subject_id=record_data.get('data_subject_id'),
                    data_categories=record_data.get('data_categories', []),
                    processing_purposes=record_data.get('processing_purposes', []),
                    legal_basis=record_data.get('legal_basis'),
                    consent_given=record_data.get('consent_given', False),
                    consent_timestamp=record_data.get('consent_timestamp'),
                    consent_scope=record_data.get('consent_scope'),
                    retention_period_days=record_data.get('retention_period_days'),
                    processing_location=record_data.get('processing_location'),
                    third_party_sharing=record_data.get('third_party_sharing', False),
                    third_parties=record_data.get('third_parties', []),
                    status=record_data.get('status', 'active'),
                    metadata=record_data.get('metadata')
                )
                db.add(record)
                db.commit()
                return record_id
                
        except Exception as e:
            logger.error(f"Failed to create data processing record: {e}")
            return None
    
    def get_by_subject(self, data_subject_id: str, tenant_id: str) -> List[Dict]:
        """根据数据主体获取记录"""
        if self._use_memory_storage:
            records = []
            for record in self._records.values():
                if record.get('data_subject_id') != data_subject_id:
                    continue
                if record.get('tenant_id') != tenant_id:
                    continue
                records.append(record)
            return records
        
        try:
            from backend.schemas.security_models import DataProcessingRecord
            
            with self._db_service.get_session() as db:
                records = db.query(DataProcessingRecord).filter(
                    DataProcessingRecord.data_subject_id == data_subject_id,
                    DataProcessingRecord.tenant_id == tenant_id
                ).order_by(desc(DataProcessingRecord.created_at)).all()
                return [r.to_dict() for r in records]
                
        except Exception as e:
            logger.error(f"Failed to get records by subject: {e}")
            return []
    
    def list_by_tenant(
        self,
        tenant_id: str,
        status: str = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict]:
        """列出租户的数据处理记录"""
        if self._use_memory_storage:
            records = []
            for record in self._records.values():
                if record.get('tenant_id') != tenant_id:
                    continue
                if status and record.get('status') != status:
                    continue
                records.append(record)
            records.sort(key=lambda x: x.get('created_at', datetime.min), reverse=True)
            return records[offset:offset + limit]
        
        try:
            from backend.schemas.security_models import DataProcessingRecord
            
            with self._db_service.get_session() as db:
                query = db.query(DataProcessingRecord).filter(
                    DataProcessingRecord.tenant_id == tenant_id
                )
                if status:
                    query = query.filter(DataProcessingRecord.status == status)
                
                records = query.order_by(desc(DataProcessingRecord.created_at)).offset(offset).limit(limit).all()
                return [r.to_dict() for r in records]
                
        except Exception as e:
            logger.error(f"Failed to list data processing records: {e}")
            return []


class ComplianceReportRepository:
    """合规报告仓库"""
    
    def __init__(self, db_service=None, use_memory_storage: bool = False):
        self._use_memory_storage = use_memory_storage
        self._db_service = None
        self._reports: Dict[str, Dict] = {}
        
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
        except ImportError as e:
            logger.warning(f"Database manager not available: {e}, using memory storage")
            self._use_memory_storage = True
    
    def create(self, report_data: Dict[str, Any]) -> Optional[str]:
        """创建合规报告"""
        report_id = report_data.get('report_id') or str(uuid.uuid4())
        
        if self._use_memory_storage:
            report_data['report_id'] = report_id
            report_data['created_at'] = datetime.utcnow()
            self._reports[report_id] = report_data
            return report_id
        
        try:
            from backend.schemas.security_models import ComplianceReport
            
            with self._db_service.get_session() as db:
                report = ComplianceReport(
                    report_id=report_id,
                    tenant_id=report_data.get('tenant_id'),
                    standard=report_data.get('standard'),
                    overall_level=report_data.get('overall_level'),
                    score=report_data.get('score'),
                    total_rules=report_data.get('total_rules', 0),
                    compliant_rules=report_data.get('compliant_rules', 0),
                    non_compliant_rules=report_data.get('non_compliant_rules', 0),
                    violations=report_data.get('violations', []),
                    recommendations=report_data.get('recommendations', []),
                    period_start=report_data.get('period_start'),
                    period_end=report_data.get('period_end'),
                    metadata=report_data.get('metadata')
                )
                db.add(report)
                db.commit()
                return report_id
                
        except Exception as e:
            logger.error(f"Failed to create compliance report: {e}")
            return None
    
    def get_by_id(self, report_id: str, tenant_id: str = None) -> Optional[Dict]:
        """根据ID获取报告"""
        if self._use_memory_storage:
            report = self._reports.get(report_id)
            if report and (not tenant_id or report.get('tenant_id') == tenant_id):
                return report
            return None
        
        try:
            from backend.schemas.security_models import ComplianceReport
            
            with self._db_service.get_session() as db:
                query = db.query(ComplianceReport).filter(
                    ComplianceReport.report_id == report_id
                )
                if tenant_id:
                    query = query.filter(ComplianceReport.tenant_id == tenant_id)
                
                report = query.first()
                return report.to_dict() if report else None
                
        except Exception as e:
            logger.error(f"Failed to get compliance report: {e}")
            return None
    
    def list_by_tenant(
        self,
        tenant_id: str,
        standard: str = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict]:
        """列出租户的合规报告"""
        if self._use_memory_storage:
            reports = []
            for report in self._reports.values():
                if report.get('tenant_id') != tenant_id:
                    continue
                if standard and report.get('standard') != standard:
                    continue
                reports.append(report)
            reports.sort(key=lambda x: x.get('created_at', datetime.min), reverse=True)
            return reports[offset:offset + limit]
        
        try:
            from backend.schemas.security_models import ComplianceReport
            
            with self._db_service.get_session() as db:
                query = db.query(ComplianceReport).filter(
                    ComplianceReport.tenant_id == tenant_id
                )
                if standard:
                    query = query.filter(ComplianceReport.standard == standard)
                
                reports = query.order_by(desc(ComplianceReport.created_at)).offset(offset).limit(limit).all()
                return [r.to_dict() for r in reports]
                
        except Exception as e:
            logger.error(f"Failed to list compliance reports: {e}")
            return []
    
    def get_latest_by_standard(self, tenant_id: str, standard: str) -> Optional[Dict]:
        """获取某标准的最新报告"""
        if self._use_memory_storage:
            reports = [
                r for r in self._reports.values()
                if r.get('tenant_id') == tenant_id and r.get('standard') == standard
            ]
            if not reports:
                return None
            reports.sort(key=lambda x: x.get('created_at', datetime.min), reverse=True)
            return reports[0]
        
        try:
            from backend.schemas.security_models import ComplianceReport
            
            with self._db_service.get_session() as db:
                report = db.query(ComplianceReport).filter(
                    ComplianceReport.tenant_id == tenant_id,
                    ComplianceReport.standard == standard
                ).order_by(desc(ComplianceReport.created_at)).first()
                
                return report.to_dict() if report else None
                
        except Exception as e:
            logger.error(f"Failed to get latest compliance report: {e}")
            return None


# ==================== 全局实例管理 ====================

_session_repository: Optional[UserSessionRepository] = None
_audit_log_repository: Optional[SecurityAuditLogRepository] = None
_user_role_repository: Optional[UserRoleRepository] = None
_access_policy_repository: Optional[AccessPolicyRepository] = None
_encryption_key_repository: Optional[EncryptionKeyRepository] = None
_data_processing_repository: Optional[DataProcessingRecordRepository] = None
_compliance_report_repository: Optional[ComplianceReportRepository] = None


def get_session_repository(use_memory: bool = False) -> UserSessionRepository:
    """获取会话仓库实例"""
    global _session_repository
    if _session_repository is None:
        _session_repository = UserSessionRepository(use_memory_storage=use_memory)
    return _session_repository


def get_audit_log_repository(use_memory: bool = False) -> SecurityAuditLogRepository:
    """获取审计日志仓库实例"""
    global _audit_log_repository
    if _audit_log_repository is None:
        _audit_log_repository = SecurityAuditLogRepository(use_memory_storage=use_memory)
    return _audit_log_repository


def get_user_role_repository(use_memory: bool = False) -> UserRoleRepository:
    """获取用户角色仓库实例"""
    global _user_role_repository
    if _user_role_repository is None:
        _user_role_repository = UserRoleRepository(use_memory_storage=use_memory)
    return _user_role_repository


def get_access_policy_repository(use_memory: bool = False) -> AccessPolicyRepository:
    """获取访问策略仓库实例"""
    global _access_policy_repository
    if _access_policy_repository is None:
        _access_policy_repository = AccessPolicyRepository(use_memory_storage=use_memory)
    return _access_policy_repository


def get_encryption_key_repository(use_memory: bool = False) -> EncryptionKeyRepository:
    """获取加密密钥仓库实例"""
    global _encryption_key_repository
    if _encryption_key_repository is None:
        _encryption_key_repository = EncryptionKeyRepository(use_memory_storage=use_memory)
    return _encryption_key_repository


def get_data_processing_repository(use_memory: bool = False) -> DataProcessingRecordRepository:
    """获取数据处理记录仓库实例"""
    global _data_processing_repository
    if _data_processing_repository is None:
        _data_processing_repository = DataProcessingRecordRepository(use_memory_storage=use_memory)
    return _data_processing_repository


def get_compliance_report_repository(use_memory: bool = False) -> ComplianceReportRepository:
    """获取合规报告仓库实例"""
    global _compliance_report_repository
    if _compliance_report_repository is None:
        _compliance_report_repository = ComplianceReportRepository(use_memory_storage=use_memory)
    return _compliance_report_repository


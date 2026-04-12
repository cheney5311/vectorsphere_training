"""认证数据访问层

提供认证相关的数据库操作，包括:
- 用户管理: CRUD操作、查询、搜索
- 会话管理: 创建、验证、撤销
- 登录尝试记录: 记录、统计、分析
- 安全事件: 记录、查询、统计
- 风险评估: 记录、查询
- 行为模式: 学习、查询、更新
- Agent记忆: 存储、检索、更新
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from sqlalchemy import or_, func, desc
from sqlalchemy.orm import Session

from backend.modules.database.manager import get_database_manager
from backend.schemas.auth_models import (
    User, UserSession, ApiKey, LoginAttempt, SecurityEvent,
    RiskAssessment, UserBehaviorPattern, UserSecurityProfile,
    AuthAgentMemory, AuthAgentReasoning, MFADevice, MFAVerification
)
from backend.schemas.permission_models import Role, UserRole

logger = logging.getLogger(__name__)


def get_auth_repository() -> 'AuthRepository':
    """获取认证仓库实例

    Returns:
        AuthRepository: 认证仓库实例
    """
    return AuthRepository()


class AuthRepository:
    """认证数据访问仓库

    提供认证相关的所有数据库操作。
    """

    def __init__(self):
        """初始化认证仓库"""
        self._db_manager = None

    @property
    def db_manager(self):
        """懒加载数据库管理器"""
        if self._db_manager is None:
            self._db_manager = get_database_manager()
        return self._db_manager

    def _get_session(self) -> Session:
        """获取数据库会话"""
        return self.db_manager.get_session()

    # ============================================================================
    # 用户管理
    # ============================================================================

    def create_user(self, user_data: Dict[str, Any]) -> User:
        """创建用户

        Args:
            user_data: 用户数据字典，包含:
                - username: 用户名
                - email: 邮箱
                - password_hash: 密码哈希
                - full_name: 全名（可选）
                - phone: 手机号码（可选）
                - role: 角色（可选，默认'user'）

        Returns:
            User: 创建的用户实体

        Raises:
            Exception: 数据库操作失败
        """
        session = self._get_session()
        try:
            user = User(**user_data)
            session.add(user)
            session.commit()
            session.refresh(user)
            logger.info("Created user: %s", user.username)
            return user
        except Exception as e:  # pylint: disable=broad-exception-caught  # pylint: disable=broad-exception-caught
            session.rollback()
            logger.error("Failed to create user: %s", e)
            raise

    def get_user_by_id(self, user_id: str) -> Optional[User]:
        """根据ID获取用户

        Args:
            user_id: 用户ID

        Returns:
            User: 用户实体或None
        """
        session = self._get_session()
        return session.query(User).filter(User.id == user_id).first()

    def get_user_by_username(self, username: str) -> Optional[User]:
        """根据用户名获取用户

        Args:
            username: 用户名

        Returns:
            User: 用户实体或None
        """
        session = self._get_session()
        return session.query(User).filter(User.username == username).first()

    def get_user_by_email(self, email: str) -> Optional[User]:
        """根据邮箱获取用户

        Args:
            email: 邮箱地址

        Returns:
            User: 用户实体或None
        """
        session = self._get_session()
        return session.query(User).filter(User.email == email).first()

    def get_user_by_identifier(self, identifier: str) -> Optional[User]:
        """根据用户名或邮箱获取用户

        Args:
            identifier: 用户名或邮箱

        Returns:
            User: 用户实体或None
        """
        session = self._get_session()
        return session.query(User).filter(
            or_(User.username == identifier, User.email == identifier)
        ).first()

    def update_user(self, user_id: str, update_data: Dict[str, Any]) -> Optional[User]:
        """更新用户信息

        Args:
            user_id: 用户ID
            update_data: 更新数据字典

        Returns:
            User: 更新后的用户实体或None
        """
        session = self._get_session()
        try:
            user = session.query(User).filter(User.id == user_id).first()
            if not user:
                return None

            for key, value in update_data.items():
                if hasattr(user, key):
                    setattr(user, key, value)

            user.updated_at = datetime.utcnow()
            session.commit()
            session.refresh(user)
            logger.info("Updated user: %s", user_id)
            return user
        except Exception as e:  # pylint: disable=broad-exception-caught
            session.rollback()
            logger.error("Failed to update user %s: %s", user_id, e)
            raise

    def delete_user(self, user_id: str) -> bool:
        """删除用户

        Args:
            user_id: 用户ID

        Returns:
            bool: 是否删除成功
        """
        session = self._get_session()
        try:
            user = session.query(User).filter(User.id == user_id).first()
            if not user:
                return False

            session.delete(user)
            session.commit()
            logger.info("Deleted user: %s", user_id)
            return True
        except Exception as e:  # pylint: disable=broad-exception-caught
            session.rollback()
            logger.error("Failed to delete user %s: %s", user_id, e)
            raise

    def list_users(self,
                   offset: int = 0,
                   limit: int = 100,
                   status: Optional[str] = None,
                   role: Optional[str] = None,
                   search: Optional[str] = None) -> Tuple[List[User], int]:
        """列出用户

        Args:
            offset: 偏移量
            limit: 限制数量
            status: 状态过滤
            role: 角色过滤
            search: 搜索关键词

        Returns:
            Tuple[List[User], int]: 用户列表和总数
        """
        session = self._get_session()
        query = session.query(User)

        if status:
            query = query.filter(User.status == status)
        if role:
            query = query.filter(User.role == role)
        if search:
            search_pattern = f"%{search}%"
            query = query.filter(
                or_(
                    User.username.ilike(search_pattern),
                    User.email.ilike(search_pattern),
                    User.full_name.ilike(search_pattern)
                )
            )

        total = query.count()
        users = query.order_by(desc(User.created_at)).offset(offset).limit(limit).all()

        return users, total

    def increment_failed_login(self, user_id: str) -> int:
        """增加登录失败次数

        Args:
            user_id: 用户ID

        Returns:
            int: 当前失败次数
        """
        session = self._get_session()
        try:
            user = session.query(User).filter(User.id == user_id).first()
            if not user:
                return 0

            user.failed_login_count = (user.failed_login_count or 0) + 1
            session.commit()
            return user.failed_login_count
        except Exception as e:  # pylint: disable=broad-exception-caught
            session.rollback()
            logger.error("Failed to increment failed login count: %s", e)
            return 0

    def reset_failed_login(self, user_id: str) -> None:
        """重置登录失败次数

        Args:
            user_id: 用户ID
        """
        session = self._get_session()
        try:
            user = session.query(User).filter(User.id == user_id).first()
            if user:
                user.failed_login_count = 0
                user.lockout_until = None
                session.commit()
        except Exception as e:  # pylint: disable=broad-exception-caught
            session.rollback()
            logger.error("Failed to reset failed login count: %s", e)

    def lock_user(self, user_id: str, lockout_minutes: int = 30) -> None:
        """锁定用户账户

        Args:
            user_id: 用户ID
            lockout_minutes: 锁定分钟数
        """
        session = self._get_session()
        try:
            user = session.query(User).filter(User.id == user_id).first()
            if user:
                user.status = 'locked'
                user.lockout_until = datetime.utcnow() + timedelta(minutes=lockout_minutes)
                session.commit()
                logger.info("Locked user %s for %d minutes", user_id, lockout_minutes)
        except Exception as e:  # pylint: disable=broad-exception-caught
            session.rollback()
            logger.error("Failed to lock user: %s", e)

    def unlock_user(self, user_id: str) -> None:
        """解锁用户账户

        Args:
            user_id: 用户ID
        """
        session = self._get_session()
        try:
            user = session.query(User).filter(User.id == user_id).first()
            if user:
                user.status = 'active'
                user.lockout_until = None
                user.failed_login_count = 0
                session.commit()
                logger.info("Unlocked user %s", user_id)
        except Exception as e:  # pylint: disable=broad-exception-caught
            session.rollback()
            logger.error("Failed to unlock user: %s", e)

    def update_last_login(self, user_id: str) -> None:
        """更新最后登录时间

        Args:
            user_id: 用户ID
        """
        session = self._get_session()
        try:
            user = session.query(User).filter(User.id == user_id).first()
            if user:
                user.last_login = datetime.utcnow()
                session.commit()
        except Exception as e:  # pylint: disable=broad-exception-caught
            session.rollback()
            logger.error("Failed to update last login: %s", e)

    def update_trust_score(self, user_id: str, trust_score: float) -> None:
        """更新用户信任分数

        Args:
            user_id: 用户ID
            trust_score: 新的信任分数 (0.0-1.0)
        """
        session = self._get_session()
        try:
            user = session.query(User).filter(User.id == user_id).first()
            if user:
                user.trust_score = max(0.0, min(1.0, trust_score))
                session.commit()
        except Exception as e:  # pylint: disable=broad-exception-caught
            session.rollback()
            logger.error("Failed to update trust score: %s", e)

    # ============================================================================
    # 会话管理
    # ============================================================================

    def create_session(self, session_data: Dict[str, Any]) -> UserSession:
        """创建用户会话

        Args:
            session_data: 会话数据字典，包含:
                - user_id: 用户ID
                - session_token: 会话令牌
                - access_token: 访问令牌
                - refresh_token: 刷新令牌
                - expires_at: 过期时间
                - ip_address: IP地址（可选）
                - user_agent: 用户代理（可选）
                - device_fingerprint: 设备指纹（可选）
                - location: 地理位置（可选）

        Returns:
            UserSession: 创建的会话实体
        """
        session = self._get_session()
        try:
            user_session = UserSession(**session_data)
            session.add(user_session)
            session.commit()
            session.refresh(user_session)
            logger.info("Created session for user: %s", session_data.get('user_id'))
            return user_session
        except Exception as e:  # pylint: disable=broad-exception-caught
            session.rollback()
            logger.error("Failed to create session: %s", e)
            raise

    def get_session_by_token(self, token: str) -> Optional[UserSession]:
        """根据令牌获取会话

        Args:
            token: 会话令牌或访问令牌

        Returns:
            UserSession: 会话实体或None
        """
        session = self._get_session()
        return session.query(UserSession).filter(
            or_(
                UserSession.session_token == token,
                UserSession.access_token == token,
                UserSession.refresh_token == token
            )
        ).first()

    def get_user_sessions(self, user_id: str, active_only: bool = True) -> List[UserSession]:
        """获取用户的所有会话

        Args:
            user_id: 用户ID
            active_only: 是否只返回活跃会话

        Returns:
            List[UserSession]: 会话列表
        """
        session = self._get_session()
        query = session.query(UserSession).filter(UserSession.user_id == user_id)

        if active_only:
            query = query.filter(
                UserSession.is_active == True,
                UserSession.expires_at > datetime.utcnow()
            )

        return query.order_by(desc(UserSession.created_at)).all()

    def invalidate_session(self, session_id: str) -> bool:
        """使会话失效

        Args:
            session_id: 会话ID

        Returns:
            bool: 是否成功
        """
        session = self._get_session()
        try:
            user_session = session.query(UserSession).filter(
                UserSession.id == session_id
            ).first()

            if user_session:
                user_session.is_active = False
                user_session.status = 'revoked'
                session.commit()
                return True
            return False
        except Exception as e:  # pylint: disable=broad-exception-caught
            session.rollback()
            logger.error("Failed to invalidate session: %s", e)
            return False

    def invalidate_user_sessions(self, user_id: str, exclude_session_id: Optional[str] = None) -> int:
        """使用户的所有会话失效

        Args:
            user_id: 用户ID
            exclude_session_id: 排除的会话ID

        Returns:
            int: 失效的会话数量
        """
        session = self._get_session()
        try:
            query = session.query(UserSession).filter(
                UserSession.user_id == user_id,
                UserSession.is_active == True
            )

            if exclude_session_id:
                query = query.filter(UserSession.id != exclude_session_id)

            count = query.update({
                'is_active': False,
                'status': 'revoked'
            })
            session.commit()
            logger.info("Invalidated %d sessions for user %s", count, user_id)
            return count
        except Exception as e:  # pylint: disable=broad-exception-caught
            session.rollback()
            logger.error("Failed to invalidate user sessions: %s", e)
            return 0

    def update_session_risk_score(self, session_id: str, risk_score: float) -> None:
        """更新会话风险分数

        Args:
            session_id: 会话ID
            risk_score: 风险分数
        """
        session = self._get_session()
        try:
            user_session = session.query(UserSession).filter(
                UserSession.id == session_id
            ).first()

            if user_session:
                user_session.risk_score = risk_score
                session.commit()
        except Exception as e:  # pylint: disable=broad-exception-caught
            session.rollback()
            logger.error("Failed to update session risk score: %s", e)

    def cleanup_expired_sessions(self) -> int:
        """清理过期会话

        Returns:
            int: 清理的会话数量
        """
        session = self._get_session()
        try:
            count = session.query(UserSession).filter(
                UserSession.expires_at < datetime.utcnow()
            ).update({'is_active': False, 'status': 'expired'})
            session.commit()
            logger.info("Cleaned up %d expired sessions", count)
            return count
        except Exception as e:  # pylint: disable=broad-exception-caught
            session.rollback()
            logger.error("Failed to cleanup expired sessions: %s", e)
            return 0

    # ============================================================================
    # 登录尝试记录
    # ============================================================================

    def record_login_attempt(self, attempt_data: Dict[str, Any]) -> LoginAttempt:
        """记录登录尝试

        Args:
            attempt_data: 登录尝试数据，包含:
                - user_id: 用户ID（可选）
                - username: 尝试的用户名
                - ip_address: IP地址
                - user_agent: 用户代理（可选）
                - device_fingerprint: 设备指纹（可选）
                - location: 地理位置（可选）
                - success: 是否成功
                - failure_reason: 失败原因（可选）
                - risk_score: 风险分数（可选）
                - risk_factors: 风险因素（可选）

        Returns:
            LoginAttempt: 登录尝试记录
        """
        session = self._get_session()
        try:
            attempt = LoginAttempt(**attempt_data)
            session.add(attempt)
            session.commit()
            session.refresh(attempt)
            return attempt
        except Exception as e:  # pylint: disable=broad-exception-caught
            session.rollback()
            logger.error("Failed to record login attempt: %s", e)
            raise

    def get_recent_login_attempts(self,
                                   user_id: Optional[str] = None,
                                   ip_address: Optional[str] = None,
                                   username: Optional[str] = None,
                                   hours: int = 24,
                                   limit: int = 100) -> List[LoginAttempt]:
        """获取最近的登录尝试

        Args:
            user_id: 用户ID（可选）
            ip_address: IP地址（可选）
            username: 用户名（可选）
            hours: 时间范围（小时）
            limit: 限制数量

        Returns:
            List[LoginAttempt]: 登录尝试列表
        """
        session = self._get_session()
        since = datetime.utcnow() - timedelta(hours=hours)

        query = session.query(LoginAttempt).filter(LoginAttempt.created_at >= since)

        if user_id:
            query = query.filter(LoginAttempt.user_id == user_id)
        if ip_address:
            query = query.filter(LoginAttempt.ip_address == ip_address)
        if username:
            query = query.filter(LoginAttempt.username == username)

        return query.order_by(desc(LoginAttempt.created_at)).limit(limit).all()

    def get_failed_login_count(self,
                                user_id: Optional[str] = None,
                                ip_address: Optional[str] = None,
                                minutes: int = 30) -> int:
        """获取失败登录次数

        Args:
            user_id: 用户ID（可选）
            ip_address: IP地址（可选）
            minutes: 时间范围（分钟）

        Returns:
            int: 失败次数
        """
        session = self._get_session()
        since = datetime.utcnow() - timedelta(minutes=minutes)

        # pylint: disable=not-callable
        query = session.query(func.count(LoginAttempt.id)).filter(
            LoginAttempt.created_at >= since,
            LoginAttempt.success == False
        )

        if user_id:
            query = query.filter(LoginAttempt.user_id == user_id)
        if ip_address:
            query = query.filter(LoginAttempt.ip_address == ip_address)

        return query.scalar() or 0

    def get_login_statistics(self,
                              user_id: Optional[str] = None,
                              days: int = 30) -> Dict[str, Any]:
        """获取登录统计

        Args:
            user_id: 用户ID（可选）
            days: 统计天数

        Returns:
            Dict: 统计数据
        """
        session = self._get_session()
        since = datetime.utcnow() - timedelta(days=days)

        base_query = session.query(LoginAttempt).filter(LoginAttempt.created_at >= since)

        if user_id:
            base_query = base_query.filter(LoginAttempt.user_id == user_id)

        total = base_query.count()
        successful = base_query.filter(LoginAttempt.success == True).count()
        failed = total - successful

        # 按IP统计
        # pylint: disable=not-callable
        ip_stats = session.query(
            LoginAttempt.ip_address,
            func.count(LoginAttempt.id).label('count')
        ).filter(
            LoginAttempt.created_at >= since
        )
        if user_id:
            ip_stats = ip_stats.filter(LoginAttempt.user_id == user_id)

        ip_stats = ip_stats.group_by(LoginAttempt.ip_address).order_by(
            desc('count')
        ).limit(10).all()

        return {
            'total_attempts': total,
            'successful': successful,
            'failed': failed,
            'success_rate': successful / total if total > 0 else 0,
            'top_ips': [{'ip': ip, 'count': count} for ip, count in ip_stats],
            'period_days': days
        }

    # ============================================================================
    # 安全事件
    # ============================================================================

    def record_security_event(self, event_data: Dict[str, Any]) -> SecurityEvent:
        """记录安全事件

        Args:
            event_data: 安全事件数据，包含:
                - user_id: 用户ID（可选）
                - event_type: 事件类型
                - severity: 严重程度
                - description: 描述
                - ip_address: IP地址（可选）
                - metadata: 元数据（可选）

        Returns:
            SecurityEvent: 安全事件记录
        """
        session = self._get_session()
        try:
            event = SecurityEvent(**event_data)
            session.add(event)
            session.commit()
            session.refresh(event)
            logger.info("Recorded security event: %s", event_data.get('event_type'))
            return event
        except Exception as e:  # pylint: disable=broad-exception-caught
            session.rollback()
            logger.error("Failed to record security event: %s", e)
            raise

    def get_security_events(self,
                             user_id: Optional[str] = None,
                             event_type: Optional[str] = None,
                             severity: Optional[str] = None,
                             is_resolved: Optional[bool] = None,
                             days: int = 30,
                             offset: int = 0,
                             limit: int = 100) -> Tuple[List[SecurityEvent], int]:
        """获取安全事件

        Args:
            user_id: 用户ID（可选）
            event_type: 事件类型（可选）
            severity: 严重程度（可选）
            is_resolved: 是否已解决（可选）
            days: 时间范围（天）
            offset: 偏移量
            limit: 限制数量

        Returns:
            Tuple[List[SecurityEvent], int]: 事件列表和总数
        """
        session = self._get_session()
        since = datetime.utcnow() - timedelta(days=days)

        query = session.query(SecurityEvent).filter(SecurityEvent.created_at >= since)

        if user_id:
            query = query.filter(SecurityEvent.user_id == user_id)
        if event_type:
            query = query.filter(SecurityEvent.event_type == event_type)
        if severity:
            query = query.filter(SecurityEvent.severity == severity)
        if is_resolved is not None:
            query = query.filter(SecurityEvent.is_resolved == is_resolved)

        total = query.count()
        events = query.order_by(desc(SecurityEvent.created_at)).offset(offset).limit(limit).all()

        return events, total

    def resolve_security_event(self,
                                event_id: str,
                                resolved_by: str,
                                resolution_notes: Optional[str] = None) -> bool:
        """解决安全事件

        Args:
            event_id: 事件ID
            resolved_by: 解决人ID
            resolution_notes: 解决说明

        Returns:
            bool: 是否成功
        """
        session = self._get_session()
        try:
            event = session.query(SecurityEvent).filter(SecurityEvent.id == event_id).first()
            if event:
                event.is_resolved = True
                event.resolved_at = datetime.utcnow()
                event.resolved_by = resolved_by
                event.resolution_notes = resolution_notes
                session.commit()
                return True
            return False
        except Exception as e:  # pylint: disable=broad-exception-caught
            session.rollback()
            logger.error("Failed to resolve security event: %s", e)
            return False

    def update_event_agent_analysis(self,
                                     event_id: str,
                                     analysis: Dict[str, Any],
                                     recommendation: str) -> bool:
        """更新事件的Agent分析

        Args:
            event_id: 事件ID
            analysis: 分析结果
            recommendation: 建议

        Returns:
            bool: 是否成功
        """
        session = self._get_session()
        try:
            event = session.query(SecurityEvent).filter(SecurityEvent.id == event_id).first()
            if event:
                event.agent_analysis = analysis
                event.agent_recommendation = recommendation
                session.commit()
                return True
            return False
        except Exception as e:  # pylint: disable=broad-exception-caught
            session.rollback()
            logger.error("Failed to update event agent analysis: %s", e)
            return False

    def get_security_statistics(self, days: int = 30) -> Dict[str, Any]:
        """获取安全统计

        Args:
            days: 统计天数

        Returns:
            Dict: 统计数据
        """
        session = self._get_session()
        since = datetime.utcnow() - timedelta(days=days)

        base_query = session.query(SecurityEvent).filter(SecurityEvent.created_at >= since)

        total = base_query.count()
        unresolved = base_query.filter(SecurityEvent.is_resolved == False).count()

        # 按类型统计
        # pylint: disable=not-callable
        by_type = session.query(
            SecurityEvent.event_type,
            func.count(SecurityEvent.id).label('count')
        ).filter(
            SecurityEvent.created_at >= since
        ).group_by(SecurityEvent.event_type).all()

        # 按严重程度统计
        # pylint: disable=not-callable
        by_severity = session.query(
            SecurityEvent.severity,
            func.count(SecurityEvent.id).label('count')
        ).filter(
            SecurityEvent.created_at >= since
        ).group_by(SecurityEvent.severity).all()

        return {
            'total_events': total,
            'unresolved': unresolved,
            'resolved': total - unresolved,
            'by_type': {t: c for t, c in by_type},
            'by_severity': {s: c for s, c in by_severity},
            'period_days': days
        }

    # ============================================================================
    # 风险评估
    # ============================================================================

    def record_risk_assessment(self, assessment_data: Dict[str, Any]) -> RiskAssessment:
        """记录风险评估

        Args:
            assessment_data: 风险评估数据

        Returns:
            RiskAssessment: 风险评估记录
        """
        session = self._get_session()
        try:
            assessment = RiskAssessment(**assessment_data)
            session.add(assessment)
            session.commit()
            session.refresh(assessment)
            return assessment
        except Exception as e:  # pylint: disable=broad-exception-caught
            session.rollback()
            logger.error("Failed to record risk assessment: %s", e)
            raise

    def get_user_risk_assessments(self,
                                   user_id: str,
                                   assessment_type: Optional[str] = None,
                                   days: int = 30,
                                   limit: int = 100) -> List[RiskAssessment]:
        """获取用户的风险评估记录

        Args:
            user_id: 用户ID
            assessment_type: 评估类型（可选）
            days: 时间范围（天）
            limit: 限制数量

        Returns:
            List[RiskAssessment]: 风险评估列表
        """
        session = self._get_session()
        since = datetime.utcnow() - timedelta(days=days)

        query = session.query(RiskAssessment).filter(
            RiskAssessment.user_id == user_id,
            RiskAssessment.created_at >= since
        )

        if assessment_type:
            query = query.filter(RiskAssessment.assessment_type == assessment_type)

        return query.order_by(desc(RiskAssessment.created_at)).limit(limit).all()

    def get_average_risk_score(self, user_id: str, days: int = 30) -> float:
        """获取用户的平均风险分数

        Args:
            user_id: 用户ID
            days: 时间范围（天）

        Returns:
            float: 平均风险分数
        """
        session = self._get_session()
        since = datetime.utcnow() - timedelta(days=days)

        # pylint: disable=not-callable
        result = session.query(func.avg(RiskAssessment.risk_score)).filter(
            RiskAssessment.user_id == user_id,
            RiskAssessment.created_at >= since
        ).scalar()

        return result or 0.0

    # ============================================================================
    # 用户行为模式
    # ============================================================================

    def get_or_create_behavior_pattern(self,
                                        user_id: str,
                                        pattern_type: str) -> UserBehaviorPattern:
        """获取或创建用户行为模式

        Args:
            user_id: 用户ID
            pattern_type: 模式类型

        Returns:
            UserBehaviorPattern: 行为模式
        """
        session = self._get_session()

        pattern = session.query(UserBehaviorPattern).filter(
            UserBehaviorPattern.user_id == user_id,
            UserBehaviorPattern.pattern_type == pattern_type
        ).first()

        if not pattern:
            pattern = UserBehaviorPattern(
                user_id=user_id,
                pattern_type=pattern_type,
                pattern_data={}
            )
            session.add(pattern)
            session.commit()
            session.refresh(pattern)

        return pattern

    def update_behavior_pattern(self,
                                 user_id: str,
                                 pattern_type: str,
                                 pattern_data: Dict[str, Any],
                                 confidence: Optional[float] = None) -> UserBehaviorPattern:
        """更新用户行为模式

        Args:
            user_id: 用户ID
            pattern_type: 模式类型
            pattern_data: 模式数据
            confidence: 置信度（可选）

        Returns:
            UserBehaviorPattern: 更新后的行为模式
        """
        session = self._get_session()
        try:
            pattern = self.get_or_create_behavior_pattern(user_id, pattern_type)

            pattern.pattern_data = pattern_data
            pattern.sample_count = pattern.sample_count + 1
            pattern.last_observed = datetime.utcnow()

            if confidence is not None:
                pattern.confidence = confidence

            session.commit()
            session.refresh(pattern)
            return pattern
        except Exception as e:  # pylint: disable=broad-exception-caught
            session.rollback()
            logger.error("Failed to update behavior pattern: %s", e)
            raise

    def get_user_behavior_patterns(self, user_id: str) -> List[UserBehaviorPattern]:
        """获取用户的所有行为模式

        Args:
            user_id: 用户ID

        Returns:
            List[UserBehaviorPattern]: 行为模式列表
        """
        session = self._get_session()
        return session.query(UserBehaviorPattern).filter(
            UserBehaviorPattern.user_id == user_id,
            UserBehaviorPattern.is_active == True
        ).all()

    # ============================================================================
    # 用户安全画像
    # ============================================================================

    def get_or_create_security_profile(self, user_id: str) -> UserSecurityProfile:
        """获取或创建用户安全画像

        Args:
            user_id: 用户ID

        Returns:
            UserSecurityProfile: 安全画像
        """
        session = self._get_session()

        profile = session.query(UserSecurityProfile).filter(
            UserSecurityProfile.user_id == user_id
        ).first()

        if not profile:
            profile = UserSecurityProfile(user_id=user_id)
            session.add(profile)
            session.commit()
            session.refresh(profile)

        return profile

    def update_security_profile(self,
                                 user_id: str,
                                 update_data: Dict[str, Any]) -> UserSecurityProfile:
        """更新用户安全画像

        Args:
            user_id: 用户ID
            update_data: 更新数据

        Returns:
            UserSecurityProfile: 更新后的安全画像
        """
        session = self._get_session()
        try:
            profile = self.get_or_create_security_profile(user_id)

            for key, value in update_data.items():
                if hasattr(profile, key):
                    setattr(profile, key, value)

            session.commit()
            session.refresh(profile)
            return profile
        except Exception as e:  # pylint: disable=broad-exception-caught
            session.rollback()
            logger.error("Failed to update security profile: %s", e)
            raise

    def add_known_ip(self, user_id: str, ip_address: str) -> None:
        """添加已知IP

        Args:
            user_id: 用户ID
            ip_address: IP地址
        """
        session = self._get_session()
        try:
            profile = self.get_or_create_security_profile(user_id)

            known_ips = profile.known_ips or []
            if ip_address not in known_ips:
                known_ips.append(ip_address)
                # 保留最近的50个IP
                profile.known_ips = known_ips[-50:]
                session.commit()
        except Exception as e:  # pylint: disable=broad-exception-caught
            session.rollback()
            logger.error("Failed to add known IP: %s", e)

    def is_known_ip(self, user_id: str, ip_address: str) -> bool:
        """检查是否为已知IP

        Args:
            user_id: 用户ID
            ip_address: IP地址

        Returns:
            bool: 是否已知
        """
        session = self._get_session()
        profile = session.query(UserSecurityProfile).filter(
            UserSecurityProfile.user_id == user_id
        ).first()

        if profile and profile.known_ips:
            return ip_address in profile.known_ips
        return False

    # ============================================================================
    # Agent 记忆
    # ============================================================================

    def add_auth_memory(self, memory_data: Dict[str, Any]) -> AuthAgentMemory:
        """添加认证Agent记忆

        Args:
            memory_data: 记忆数据，包含:
                - user_id: 用户ID（可选）
                - memory_type: 记忆类型
                - content: 记忆内容
                - importance: 重要性（可选）
                - metadata: 元数据（可选）

        Returns:
            AuthAgentMemory: 记忆记录
        """
        session = self._get_session()
        try:
            memory = AuthAgentMemory(**memory_data)
            session.add(memory)
            session.commit()
            session.refresh(memory)
            return memory
        except Exception as e:  # pylint: disable=broad-exception-caught
            session.rollback()
            logger.error("Failed to add auth memory: %s", e)
            raise

    def get_auth_memories(self,
                          user_id: Optional[str] = None,
                          memory_type: Optional[str] = None,
                          min_importance: float = 0.0,
                          limit: int = 100) -> List[AuthAgentMemory]:
        """获取认证Agent记忆

        Args:
            user_id: 用户ID（可选）
            memory_type: 记忆类型（可选）
            min_importance: 最小重要性
            limit: 限制数量

        Returns:
            List[AuthAgentMemory]: 记忆列表
        """
        session = self._get_session()

        query = session.query(AuthAgentMemory).filter(
            AuthAgentMemory.is_active == True,
            AuthAgentMemory.importance >= min_importance
        )

        if user_id:
            query = query.filter(
                or_(
                    AuthAgentMemory.user_id == user_id,
                    AuthAgentMemory.user_id.is_(None)
                )
            )

        if memory_type:
            query = query.filter(AuthAgentMemory.memory_type == memory_type)

        # 过滤过期记忆
        query = query.filter(
            or_(
                AuthAgentMemory.expires_at.is_(None),
                AuthAgentMemory.expires_at > datetime.utcnow()
            )
        )

        return query.order_by(desc(AuthAgentMemory.importance)).limit(limit).all()

    def search_auth_memories(self,
                              query_text: str,
                              user_id: Optional[str] = None,
                              limit: int = 10) -> List[AuthAgentMemory]:
        """搜索认证Agent记忆

        Args:
            query_text: 搜索文本
            user_id: 用户ID（可选）
            limit: 限制数量

        Returns:
            List[AuthAgentMemory]: 匹配的记忆列表
        """
        session = self._get_session()

        search_pattern = f"%{query_text}%"
        query = session.query(AuthAgentMemory).filter(
            AuthAgentMemory.is_active == True,
            AuthAgentMemory.content.ilike(search_pattern)
        )

        if user_id:
            query = query.filter(
                or_(
                    AuthAgentMemory.user_id == user_id,
                    AuthAgentMemory.user_id.is_(None)
                )
            )

        memories = query.order_by(desc(AuthAgentMemory.importance)).limit(limit).all()

        # 更新访问计数
        for memory in memories:
            memory.access_count += 1
            memory.last_accessed = datetime.utcnow()

        try:
            session.commit()
        except Exception:  # pylint: disable=broad-exception-caught
            session.rollback()
        
        return memories

    def update_memory_importance(self, memory_id: str, importance: float) -> bool:
        """更新记忆重要性

        Args:
            memory_id: 记忆ID
            importance: 新的重要性分数

        Returns:
            bool: 是否成功
        """
        session = self._get_session()
        try:
            memory = session.query(AuthAgentMemory).filter(
                AuthAgentMemory.id == memory_id
            ).first()

            if memory:
                memory.importance = max(0.0, min(1.0, importance))
                session.commit()
                return True
            return False
        except Exception as e:  # pylint: disable=broad-exception-caught
            session.rollback()
            logger.error("Failed to update memory importance: %s", e)
            return False

    def deactivate_memory(self, memory_id: str) -> bool:
        """停用记忆

        Args:
            memory_id: 记忆ID

        Returns:
            bool: 是否成功
        """
        session = self._get_session()
        try:
            memory = session.query(AuthAgentMemory).filter(
                AuthAgentMemory.id == memory_id
            ).first()

            if memory:
                memory.is_active = False
                session.commit()
                return True
            return False
        except Exception as e:  # pylint: disable=broad-exception-caught
            session.rollback()
            logger.error("Failed to deactivate memory: %s", e)
            return False

    # ============================================================================
    # Agent 推理记录
    # ============================================================================

    def record_reasoning(self, reasoning_data: Dict[str, Any]) -> AuthAgentReasoning:
        """记录Agent推理

        Args:
            reasoning_data: 推理数据

        Returns:
            AuthAgentReasoning: 推理记录
        """
        session = self._get_session()
        try:
            reasoning = AuthAgentReasoning(**reasoning_data)
            session.add(reasoning)
            session.commit()
            session.refresh(reasoning)
            return reasoning
        except Exception as e:  # pylint: disable=broad-exception-caught
            session.rollback()
            logger.error("Failed to record reasoning: %s", e)
            raise

    def get_reasoning_history(self,
                               user_id: Optional[str] = None,
                               trigger: Optional[str] = None,
                               days: int = 30,
                               limit: int = 100) -> List[AuthAgentReasoning]:
        """获取推理历史

        Args:
            user_id: 用户ID（可选）
            trigger: 触发事件（可选）
            days: 时间范围（天）
            limit: 限制数量

        Returns:
            List[AuthAgentReasoning]: 推理记录列表
        """
        session = self._get_session()
        since = datetime.utcnow() - timedelta(days=days)

        query = session.query(AuthAgentReasoning).filter(
            AuthAgentReasoning.created_at >= since
        )

        if user_id:
            query = query.filter(AuthAgentReasoning.user_id == user_id)
        if trigger:
            query = query.filter(AuthAgentReasoning.trigger == trigger)

        return query.order_by(desc(AuthAgentReasoning.created_at)).limit(limit).all()

    def update_reasoning_outcome(self,
                                  reasoning_id: str,
                                  outcome: str,
                                  feedback: Optional[Dict[str, Any]] = None) -> bool:
        """更新推理结果

        Args:
            reasoning_id: 推理记录ID
            outcome: 结果
            feedback: 反馈

        Returns:
            bool: 是否成功
        """
        session = self._get_session()
        try:
            reasoning = session.query(AuthAgentReasoning).filter(
                AuthAgentReasoning.id == reasoning_id
            ).first()

            if reasoning:
                reasoning.outcome = outcome
                if feedback:
                    reasoning.feedback = feedback
                session.commit()
                return True
            return False
        except Exception as e:  # pylint: disable=broad-exception-caught
            session.rollback()
            logger.error("Failed to update reasoning outcome: %s", e)
            return False

    # ============================================================================
    # MFA设备管理
    # ============================================================================

    def add_mfa_device(self, device_data: Dict[str, Any]) -> MFADevice:
        """添加MFA设备

        Args:
            device_data: 设备数据

        Returns:
            MFADevice: MFA设备记录
        """
        session = self._get_session()
        try:
            device = MFADevice(**device_data)
            session.add(device)
            session.commit()
            session.refresh(device)
            return device
        except Exception as e:  # pylint: disable=broad-exception-caught
            session.rollback()
            logger.error("Failed to add MFA device: %s", e)
            raise

    def get_user_mfa_devices(self, user_id: str, active_only: bool = True) -> List[MFADevice]:
        """获取用户的MFA设备

        Args:
            user_id: 用户ID
            active_only: 是否只返回激活的设备

        Returns:
            List[MFADevice]: MFA设备列表
        """
        session = self._get_session()
        query = session.query(MFADevice).filter(MFADevice.user_id == user_id)

        if active_only:
            query = query.filter(MFADevice.is_active == True)

        return query.all()

    def verify_mfa_device(self, device_id: str) -> bool:
        """验证MFA设备

        Args:
            device_id: 设备ID

        Returns:
            bool: 是否成功
        """
        session = self._get_session()
        try:
            device = session.query(MFADevice).filter(MFADevice.id == device_id).first()
            if device:
                device.is_verified = True
                device.verified_at = datetime.utcnow()
                session.commit()
                return True
            return False
        except Exception as e:  # pylint: disable=broad-exception-caught
            session.rollback()
            logger.error("Failed to verify MFA device: %s", e)
            return False

    def record_mfa_verification(self, verification_data: Dict[str, Any]) -> MFAVerification:
        """记录MFA验证

        Args:
            verification_data: 验证数据

        Returns:
            MFAVerification: 验证记录
        """
        session = self._get_session()
        try:
            verification = MFAVerification(**verification_data)
            session.add(verification)
            session.commit()
            session.refresh(verification)
            return verification
        except Exception as e:  # pylint: disable=broad-exception-caught
            session.rollback()
            logger.error("Failed to record MFA verification: %s", e)
            raise

    # ============================================================================
    # 角色权限管理
    # ============================================================================

    def get_role_by_name(self, name: str) -> Optional[Role]:
        """根据名称获取角色

        Args:
            name: 角色名称

        Returns:
            Role: 角色或None
        """
        session = self._get_session()
        return session.query(Role).filter(Role.name == name).first()

    def get_user_roles(self, user_id: str) -> List[Role]:
        """获取用户的角色

        Args:
            user_id: 用户ID

        Returns:
            List[Role]: 角色列表
        """
        session = self._get_session()
        user_roles = session.query(UserRole).filter(
            UserRole.user_id == user_id,
            UserRole.is_active == True
        ).all()

        role_ids = [ur.role_id for ur in user_roles]
        if not role_ids:
            return []

        return session.query(Role).filter(
            Role.id.in_(role_ids),
            Role.is_active == True
        ).all()

    def assign_role(self, user_id: str, role_id: str, granted_by: Optional[str] = None) -> UserRole:
        """分配角色给用户

        Args:
            user_id: 用户ID
            role_id: 角色ID
            granted_by: 授权人ID

        Returns:
            UserRole: 用户角色关联
        """
        session = self._get_session()
        try:
            # 检查是否已存在
            existing = session.query(UserRole).filter(
                UserRole.user_id == user_id,
                UserRole.role_id == role_id
            ).first()

            if existing:
                existing.is_active = True
                session.commit()
                return existing

            user_role = UserRole(
                user_id=user_id,
                role_id=role_id,
                granted_by=granted_by
            )
            session.add(user_role)
            session.commit()
            session.refresh(user_role)
            return user_role
        except Exception as e:  # pylint: disable=broad-exception-caught
            session.rollback()
            logger.error("Failed to assign role: %s", e)
            raise

    def revoke_role(self, user_id: str, role_id: str) -> bool:
        """撤销用户角色

        Args:
            user_id: 用户ID
            role_id: 角色ID

        Returns:
            bool: 是否成功
        """
        session = self._get_session()
        try:
            user_role = session.query(UserRole).filter(
                UserRole.user_id == user_id,
                UserRole.role_id == role_id
            ).first()

            if user_role:
                user_role.is_active = False
                session.commit()
                return True
            return False
        except Exception as e:  # pylint: disable=broad-exception-caught
            session.rollback()
            logger.error("Failed to revoke role: %s", e)
            return False

    def has_permission(self, user_id: str, permission: str) -> bool:
        """检查用户是否有权限

        Args:
            user_id: 用户ID
            permission: 权限名称

        Returns:
            bool: 是否有权限
        """
        roles = self.get_user_roles(user_id)

        for role in roles:
            if role.permissions and permission in role.permissions:
                return True

        return False

    # ============================================================================
    # API密钥管理
    # ============================================================================

    def create_api_key(self, key_data: Dict[str, Any]) -> ApiKey:
        """创建API密钥

        Args:
            key_data: 密钥数据

        Returns:
            ApiKey: API密钥记录
        """
        session = self._get_session()
        try:
            api_key = ApiKey(**key_data)
            session.add(api_key)
            session.commit()
            session.refresh(api_key)
            return api_key
        except Exception as e:  # pylint: disable=broad-exception-caught
            session.rollback()
            logger.error("Failed to create API key: %s", e)
            raise

    def get_api_key_by_hash(self, key_hash: str) -> Optional[ApiKey]:
        """根据哈希获取API密钥

        Args:
            key_hash: 密钥哈希

        Returns:
            ApiKey: API密钥或None
        """
        session = self._get_session()
        return session.query(ApiKey).filter(
            ApiKey.key_hash == key_hash,
            ApiKey.is_active == True
        ).first()

    def get_user_api_keys(self, user_id: str) -> List[ApiKey]:
        """获取用户的API密钥

        Args:
            user_id: 用户ID

        Returns:
            List[ApiKey]: API密钥列表
        """
        session = self._get_session()
        return session.query(ApiKey).filter(
            ApiKey.user_id == user_id
        ).order_by(desc(ApiKey.created_at)).all()

    def update_api_key_usage(self, key_id: str) -> None:
        """更新API密钥使用记录

        Args:
            key_id: 密钥ID
        """
        session = self._get_session()
        try:
            api_key = session.query(ApiKey).filter(ApiKey.id == key_id).first()
            if api_key:
                api_key.last_used_at = datetime.utcnow()
                api_key.usage_count = (api_key.usage_count or 0) + 1
                session.commit()
        except Exception as e:  # pylint: disable=broad-exception-caught
            session.rollback()
            logger.error("Failed to update API key usage: %s", e)

    def revoke_api_key(self, key_id: str) -> bool:
        """撤销API密钥

        Args:
            key_id: 密钥ID

        Returns:
            bool: 是否成功
        """
        session = self._get_session()
        try:
            api_key = session.query(ApiKey).filter(ApiKey.id == key_id).first()
            if api_key:
                api_key.is_active = False
                session.commit()
                return True
            return False
        except Exception as e:  # pylint: disable=broad-exception-caught
            session.rollback()
            logger.error("Failed to revoke API key: %s", e)
            return False

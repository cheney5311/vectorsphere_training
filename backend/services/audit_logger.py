# -*- coding: utf-8 -*-
"""
审计日志服务
"""

import hashlib
import json
import logging
import os
import sqlite3
import threading
from datetime import datetime, timedelta
from queue import Queue
from typing import Dict, List, Optional, Any

from backend.modules.security.models import AuditEvent, AuditQuery, AuditEventType, AuditLevel


class AuditStorage:
    """审计存储接口"""
    
    def store_event(self, event: AuditEvent) -> bool:
        """存储审计事件"""
        raise NotImplementedError
    
    def query_events(self, query: AuditQuery) -> List[AuditEvent]:
        """查询审计事件"""
        raise NotImplementedError
    
    def get_event_count(self, query: AuditQuery) -> int:
        """获取事件数量"""
        raise NotImplementedError
    
    def cleanup_old_events(self, retention_days: int) -> int:
        """清理旧事件"""
        raise NotImplementedError


class SQLiteAuditStorage(AuditStorage):
    """SQLite审计存储实现"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """初始化数据库"""
        # 确保目录存在
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS audit_events (
                    id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    level TEXT NOT NULL,
                    user_id TEXT,
                    session_id TEXT,
                    source_ip TEXT,
                    user_agent TEXT,
                    resource TEXT,
                    action TEXT,
                    result TEXT NOT NULL,
                    message TEXT NOT NULL,
                    details TEXT,
                    risk_score REAL NOT NULL,
                    tags TEXT
                )
            """)
            
            # 创建索引
            conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON audit_events(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_event_type ON audit_events(event_type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_user_id ON audit_events(user_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_level ON audit_events(level)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_risk_score ON audit_events(risk_score)")
    
    def store_event(self, event: AuditEvent) -> bool:
        """存储审计事件"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO audit_events (
                        id, timestamp, event_type, level, user_id, session_id,
                        source_ip, user_agent, resource, action, result,
                        message, details, risk_score, tags
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    event.id,
                    event.timestamp.isoformat(),
                    event.event_type.value,
                    event.level.value,
                    event.user_id,
                    event.session_id,
                    event.source_ip,
                    event.user_agent,
                    event.resource,
                    event.action,
                    event.result,
                    event.message,
                    json.dumps(event.details),
                    event.risk_score,
                    json.dumps(event.tags)
                ))
            return True
        except Exception as e:
            logging.error(f"Failed to store audit event: {e}")
            return False
    
    def query_events(self, query: AuditQuery) -> List[AuditEvent]:
        """查询审计事件"""
        sql = "SELECT * FROM audit_events WHERE 1=1"
        params = []
        
        if query.start_time:
            sql += " AND timestamp >= ?"
            params.append(query.start_time.isoformat())
        
        if query.end_time:
            sql += " AND timestamp <= ?"
            params.append(query.end_time.isoformat())
        
        if query.event_types:
            placeholders = ",".join(["?"] * len(query.event_types))
            sql += f" AND event_type IN ({placeholders})"
            params.extend([et.value for et in query.event_types])
        
        if query.levels:
            placeholders = ",".join(["?"] * len(query.levels))
            sql += f" AND level IN ({placeholders})"
            params.extend([level.value for level in query.levels])
        
        if query.user_ids:
            placeholders = ",".join(["?"] * len(query.user_ids))
            sql += f" AND user_id IN ({placeholders})"
            params.extend(query.user_ids)
        
        if query.results:
            placeholders = ",".join(["?"] * len(query.results))
            sql += f" AND result IN ({placeholders})"
            params.extend(query.results)
        
        if query.min_risk_score is not None:
            sql += " AND risk_score >= ?"
            params.append(query.min_risk_score)
        
        if query.max_risk_score is not None:
            sql += " AND risk_score <= ?"
            params.append(query.max_risk_score)
        
        sql += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
        params.extend([query.limit, query.offset])
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(sql, params)
                rows = cursor.fetchall()
                
                events = []
                for row in rows:
                    event = AuditEvent(
                        id=row['id'],
                        timestamp=datetime.fromisoformat(row['timestamp']),
                        event_type=AuditEventType(row['event_type']),
                        level=AuditLevel(row['level']),
                        user_id=row['user_id'],
                        session_id=row['session_id'],
                        source_ip=row['source_ip'],
                        user_agent=row['user_agent'],
                        resource=row['resource'],
                        action=row['action'],
                        result=row['result'],
                        message=row['message'],
                        details=json.loads(row['details'] or '{}'),
                        risk_score=row['risk_score'],
                        tags=json.loads(row['tags'] or '[]')
                    )
                    events.append(event)
                
                return events
        except Exception as e:
            logging.error(f"Failed to query audit events: {e}")
            return []
    
    def get_event_count(self, query: AuditQuery) -> int:
        """获取事件数量"""
        sql = "SELECT COUNT(*) FROM audit_events WHERE 1=1"
        params = []
        
        # 添加查询条件（与query_events相同的逻辑）
        if query.start_time:
            sql += " AND timestamp >= ?"
            params.append(query.start_time.isoformat())
        
        if query.end_time:
            sql += " AND timestamp <= ?"
            params.append(query.end_time.isoformat())
        
        # ... 其他条件类似
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(sql, params)
                return cursor.fetchone()[0]
        except Exception as e:
            logging.error(f"Failed to get event count: {e}")
            return 0
    
    def cleanup_old_events(self, retention_days: int) -> int:
        """清理旧事件"""
        cutoff_date = datetime.now() - timedelta(days=retention_days)
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "DELETE FROM audit_events WHERE timestamp < ?",
                    (cutoff_date.isoformat(),)
                )
                return cursor.rowcount
        except Exception as e:
            logging.error(f"Failed to cleanup old events: {e}")
            return 0


class AuditLogger:
    """审计日志记录器
    
    提供全面的安全审计功能，包括事件记录、查询、分析等
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # 初始化存储
        storage_type = config.get('storage_type', 'sqlite')
        if storage_type == 'sqlite':
            db_path = config.get('db_path', 'data/audit.db')
            self.storage = SQLiteAuditStorage(db_path)
        else:
            raise ValueError(f"Unsupported storage type: {storage_type}")
        
        # 异步处理队列
        self.event_queue = Queue()
        self.processing_thread = None
        self.running = False
        
        # 风险评分配置
        self.risk_weights = config.get('risk_weights', {
            'failed_login': 0.3,
            'access_denied': 0.2,
            'suspicious_activity': 0.8,
            'policy_violation': 0.6,
            'data_export': 0.4,
            'admin_action': 0.5
        })
        
        # 启动处理线程
        self.start_processing()
    
    def start_processing(self):
        """启动事件处理线程"""
        if not self.running:
            self.running = True
            self.processing_thread = threading.Thread(target=self._process_events)
            self.processing_thread.daemon = True
            self.processing_thread.start()
    
    def stop_processing(self):
        """停止事件处理线程"""
        self.running = False
        if self.processing_thread:
            self.processing_thread.join(timeout=5)
    
    def log_event(self, event_type: AuditEventType, message: str,
                  user_id: Optional[str] = None,
                  session_id: Optional[str] = None,
                  source_ip: Optional[str] = None,
                  user_agent: Optional[str] = None,
                  resource: Optional[str] = None,
                  action: Optional[str] = None,
                  result: str = "success",
                  details: Optional[Dict[str, Any]] = None,
                  level: Optional[AuditLevel] = None,
                  tags: Optional[List[str]] = None) -> str:
        """记录审计事件
        
        Args:
            event_type: 事件类型
            message: 事件消息
            user_id: 用户ID
            session_id: 会话ID
            source_ip: 源IP地址
            user_agent: 用户代理
            resource: 资源
            action: 操作
            result: 结果
            details: 详细信息
            level: 审计级别
            tags: 标签
            
        Returns:
            事件ID
        """
        if details is None:
            details = {}
        if tags is None:
            tags = []
        
        # 生成事件ID
        event_id = self._generate_event_id(event_type, user_id, resource)
        
        # 自动确定审计级别
        if level is None:
            level = self._determine_audit_level(event_type, result, details)
        
        # 计算风险评分
        risk_score = self._calculate_risk_score(event_type, result, details, user_id)
        
        # 创建审计事件
        event = AuditEvent(
            id=event_id,
            timestamp=datetime.now(),
            event_type=event_type,
            level=level,
            user_id=user_id,
            session_id=session_id,
            source_ip=source_ip,
            user_agent=user_agent,
            resource=resource,
            action=action,
            result=result,
            message=message,
            details=details,
            risk_score=risk_score,
            tags=tags
        )
        
        # 添加到处理队列
        self.event_queue.put(event)
        
        return event_id
    
    def query_events(self, query: AuditQuery) -> List[AuditEvent]:
        """查询审计事件
        
        Args:
            query: 查询条件
            
        Returns:
            事件列表
        """
        return self.storage.query_events(query)
    
    def get_event_statistics(self, start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        """获取事件统计信息
        
        Args:
            start_time: 开始时间
            end_time: 结束时间
            
        Returns:
            统计信息
        """
        query = AuditQuery(start_time=start_time, end_time=end_time, limit=10000)
        events = self.storage.query_events(query)
        
        stats = {
            'total_events': len(events),
            'event_types': {},
            'levels': {},
            'results': {},
            'users': {},
            'risk_distribution': {
                'low': 0,
                'medium': 0,
                'high': 0,
                'critical': 0
            },
            'hourly_distribution': {},
            'top_resources': {},
            'failed_attempts': 0,
            'security_events': 0
        }
        
        for event in events:
            # 事件类型统计
            event_type = event.event_type.value
            stats['event_types'][event_type] = stats['event_types'].get(event_type, 0) + 1
            
            # 级别统计
            level = event.level.value
            stats['levels'][level] = stats['levels'].get(level, 0) + 1
            
            # 结果统计
            result = event.result
            stats['results'][result] = stats['results'].get(result, 0) + 1
            
            # 用户统计
            if event.user_id:
                stats['users'][event.user_id] = stats['users'].get(event.user_id, 0) + 1
            
            # 风险分布
            if event.risk_score < 0.3:
                stats['risk_distribution']['low'] += 1
            elif event.risk_score < 0.6:
                stats['risk_distribution']['medium'] += 1
            elif event.risk_score < 0.8:
                stats['risk_distribution']['high'] += 1
            else:
                stats['risk_distribution']['critical'] += 1
            
            # 时间分布
            hour = event.timestamp.strftime('%H')
            stats['hourly_distribution'][hour] = stats['hourly_distribution'].get(hour, 0) + 1
            
            # 资源统计
            if event.resource:
                stats['top_resources'][event.resource] = stats['top_resources'].get(event.resource, 0) + 1
            
            # 失败尝试
            if event.result == 'failure':
                stats['failed_attempts'] += 1
            
            # 安全事件
            if 'security' in event.event_type.value:
                stats['security_events'] += 1
        
        return stats
    
    def detect_anomalies(self, user_id: str, time_window_hours: int = 24) -> List[Dict[str, Any]]:
        """检测异常行为
        
        Args:
            user_id: 用户ID
            time_window_hours: 时间窗口（小时）
            
        Returns:
            异常列表
        """
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=time_window_hours)
        
        query = AuditQuery(
            start_time=start_time,
            end_time=end_time,
            user_ids=[user_id],
            limit=10000
        )
        
        events = self.storage.query_events(query)
        anomalies = []
        
        # 检测异常登录时间
        login_events = [e for e in events if e.event_type == AuditEventType.LOGIN_SUCCESS]
        unusual_hours = []
        for event in login_events:
            hour = event.timestamp.hour
            if hour < 6 or hour > 22:  # 非工作时间
                unusual_hours.append(event)
        
        if unusual_hours:
            anomalies.append({
                'type': 'unusual_login_time',
                'severity': 'medium',
                'count': len(unusual_hours),
                'description': f'User logged in during unusual hours {len(unusual_hours)} times'
            })
        
        # 检测频繁失败尝试
        failed_events = [e for e in events if e.result == 'failure']
        if len(failed_events) > 10:
            anomalies.append({
                'type': 'frequent_failures',
                'severity': 'high',
                'count': len(failed_events),
                'description': f'User had {len(failed_events)} failed attempts'
            })
        
        # 检测高风险操作
        high_risk_events = [e for e in events if e.risk_score > 0.7]
        if high_risk_events:
            anomalies.append({
                'type': 'high_risk_operations',
                'severity': 'high',
                'count': len(high_risk_events),
                'description': f'User performed {len(high_risk_events)} high-risk operations'
            })
        
        return anomalies
    
    def cleanup_old_events(self, retention_days: int = 90) -> int:
        """清理旧事件
        
        Args:
            retention_days: 保留天数
            
        Returns:
            清理的事件数量
        """
        return self.storage.cleanup_old_events(retention_days)
    
    def _process_events(self):
        """处理事件队列"""
        while self.running:
            try:
                event = self.event_queue.get(timeout=1)
                self.storage.store_event(event)
                
                # 检查是否需要实时告警
                self._check_real_time_alerts(event)
                
            except Exception as e:
                if self.running:  # 只在运行时记录错误
                    logging.error(f"Error processing audit event: {e}")
    
    def _check_real_time_alerts(self, event: AuditEvent):
        """检查实时告警"""
        # 高风险事件立即告警
        if event.risk_score > 0.8:
            self._send_alert(event, "High risk event detected")
        
        # 安全违规事件
        if event.event_type in [AuditEventType.SECURITY_VIOLATION, 
                               AuditEventType.INTRUSION_ATTEMPT]:
            self._send_alert(event, "Security violation detected")
        
        # 连续失败登录
        if event.event_type == AuditEventType.LOGIN_FAILURE:
            self._check_brute_force_attack(event)
    
    def _send_alert(self, event: AuditEvent, alert_message: str):
        """发送告警"""
        # 这里应该集成实际的告警系统
        logging.warning(f"SECURITY ALERT: {alert_message} - Event: {event.id}")
    
    def _check_brute_force_attack(self, event: AuditEvent):
        """检查暴力破解攻击"""
        if not event.user_id:
            return
        
        # 检查最近5分钟的失败登录次数
        start_time = datetime.now() - timedelta(minutes=5)
        query = AuditQuery(
            start_time=start_time,
            event_types=[AuditEventType.LOGIN_FAILURE],
            user_ids=[event.user_id],
            limit=100
        )
        
        recent_failures = self.storage.query_events(query)
        if len(recent_failures) >= 5:
            self._send_alert(event, f"Possible brute force attack on user {event.user_id}")
    
    def _generate_event_id(self, event_type: AuditEventType, 
                          user_id: Optional[str], resource: Optional[str]) -> str:
        """生成事件ID"""
        timestamp = datetime.now().isoformat()
        data = f"{timestamp}:{event_type.value}:{user_id}:{resource}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]
    
    def _determine_audit_level(self, event_type: AuditEventType, 
                              result: str, details: Dict[str, Any]) -> AuditLevel:
        """确定审计级别"""
        # 安全相关事件
        if 'security' in event_type.value:
            return AuditLevel.CRITICAL
        
        # 失败事件
        if result == 'failure':
            return AuditLevel.HIGH
        
        # 管理员操作
        if 'admin' in event_type.value or details.get('admin_action'):
            return AuditLevel.HIGH
        
        # 数据操作
        if event_type.value.startswith('data.'):
            return AuditLevel.MEDIUM
        
        # 默认级别
        return AuditLevel.LOW
    
    def _calculate_risk_score(self, event_type: AuditEventType, result: str,
                             details: Dict[str, Any], user_id: Optional[str]) -> float:
        """计算风险评分"""
        base_score = 0.1
        
        # 事件类型权重
        if 'security' in event_type.value:
            base_score += 0.7
        elif 'failure' in event_type.value or result == 'failure':
            base_score += 0.4
        elif event_type.value.startswith('data.delete'):
            base_score += 0.5
        elif event_type.value.startswith('admin.'):
            base_score += 0.3
        
        # 结果权重
        if result == 'failure':
            base_score += 0.3
        elif result == 'error':
            base_score += 0.2
        
        # 详细信息权重
        if details.get('sensitive_data'):
            base_score += 0.3
        if details.get('external_access'):
            base_score += 0.2
        if details.get('bulk_operation'):
            base_score += 0.2
        
        # 确保分数在0-1范围内
        return min(1.0, max(0.0, base_score))
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""文件访问控制与权限管理服务

提供细粒度的文件访问控制和权限管理功能。
"""

import json
import logging
import os
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple

from backend.services.access_control import AccessControlManager
from backend.services.artifact_security import SecurityLevel

logger = logging.getLogger(__name__)


class AccessType(Enum):
    """访问类型"""
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    EXECUTE = "execute"
    ADMIN = "admin"


class PermissionResult(Enum):
    """权限检查结果"""
    GRANTED = "granted"
    DENIED = "denied"
    CONDITIONAL = "conditional"


@dataclass
class FilePermission:
    """文件权限"""
    file_path: str
    user_id: str
    access_types: List[AccessType]
    granted_by: str
    granted_at: datetime
    expires_at: Optional[datetime] = None
    conditions: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class AccessRequest:
    """访问请求"""
    user_id: str
    file_path: str
    access_type: AccessType
    request_time: datetime
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class AccessLog:
    """访问日志"""
    id: str
    user_id: str
    file_path: str
    access_type: AccessType
    result: PermissionResult
    timestamp: datetime
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    reason: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class FileAccessControlService:
    """文件访问控制服务"""
    
    def __init__(self, config: Dict[str, Any]):
        """初始化服务
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.storage_path = config.get('file_access_control', {}).get('storage_path', '/tmp/file_access_control')
        self.permissions_file = os.path.join(self.storage_path, 'permissions.json')
        self.access_logs_file = os.path.join(self.storage_path, 'access_logs.json')
        
        # 创建存储目录
        os.makedirs(self.storage_path, exist_ok=True)
        
        # 初始化访问控制管理器
        self.access_control = AccessControlManager(config)
        
        # 加载权限和日志
        self.permissions: Dict[str, FilePermission] = self._load_permissions()
        self.access_logs: List[AccessLog] = self._load_access_logs()
        
        # 默认权限配置
        self.default_permissions = self._get_default_permissions()
        
        logger.info("文件访问控制服务初始化完成")
    
    def _load_permissions(self) -> Dict[str, FilePermission]:
        """加载权限数据"""
        try:
            if os.path.exists(self.permissions_file):
                with open(self.permissions_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    permissions = {}
                    for key, perm_data in data.items():
                        perm_data['access_types'] = [AccessType(at) for at in perm_data['access_types']]
                        perm_data['granted_at'] = datetime.fromisoformat(perm_data['granted_at'])
                        if perm_data.get('expires_at'):
                            perm_data['expires_at'] = datetime.fromisoformat(perm_data['expires_at'])
                        permissions[key] = FilePermission(**perm_data)
                    return permissions
        except Exception as e:
            logger.error(f"加载权限数据失败: {e}")
        
        return {}
    
    def _save_permissions(self):
        """保存权限数据"""
        try:
            data = {}
            for key, permission in self.permissions.items():
                perm_dict = asdict(permission)
                perm_dict['access_types'] = [at.value for at in permission.access_types]
                perm_dict['granted_at'] = permission.granted_at.isoformat()
                if permission.expires_at:
                    perm_dict['expires_at'] = permission.expires_at.isoformat()
                data[key] = perm_dict
            
            with open(self.permissions_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            logger.error(f"保存权限数据失败: {e}")
    
    def _load_access_logs(self) -> List[AccessLog]:
        """加载访问日志"""
        try:
            if os.path.exists(self.access_logs_file):
                with open(self.access_logs_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    logs = []
                    for log_data in data:
                        log_data['access_type'] = AccessType(log_data['access_type'])
                        log_data['result'] = PermissionResult(log_data['result'])
                        log_data['timestamp'] = datetime.fromisoformat(log_data['timestamp'])
                        logs.append(AccessLog(**log_data))
                    return logs
        except Exception as e:
            logger.error(f"加载访问日志失败: {e}")
        
        return []
    
    def _save_access_logs(self):
        """保存访问日志"""
        try:
            # 只保留最近的日志（避免文件过大）
            max_logs = self.config.get('file_access_control', {}).get('max_logs', 10000)
            if len(self.access_logs) > max_logs:
                self.access_logs = self.access_logs[-max_logs:]
            
            data = []
            for log in self.access_logs:
                log_dict = asdict(log)
                log_dict['access_type'] = log.access_type.value
                log_dict['result'] = log.result.value
                log_dict['timestamp'] = log.timestamp.isoformat()
                data.append(log_dict)
            
            with open(self.access_logs_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            logger.error(f"保存访问日志失败: {e}")
    
    def _get_default_permissions(self) -> Dict[SecurityLevel, Dict[str, List[AccessType]]]:
        """获取默认权限配置"""
        return {
            SecurityLevel.PUBLIC: {
                'owner': [AccessType.READ, AccessType.WRITE, AccessType.DELETE, AccessType.ADMIN],
                'admin': [AccessType.READ, AccessType.WRITE, AccessType.DELETE, AccessType.ADMIN],
                'user': [AccessType.READ],
                'guest': [AccessType.READ]
            },
            SecurityLevel.INTERNAL: {
                'owner': [AccessType.READ, AccessType.WRITE, AccessType.DELETE, AccessType.ADMIN],
                'admin': [AccessType.READ, AccessType.WRITE, AccessType.DELETE, AccessType.ADMIN],
                'user': [AccessType.READ, AccessType.WRITE],
                'guest': []
            },
            SecurityLevel.CONFIDENTIAL: {
                'owner': [AccessType.READ, AccessType.WRITE, AccessType.DELETE, AccessType.ADMIN],
                'admin': [AccessType.READ, AccessType.WRITE, AccessType.DELETE, AccessType.ADMIN],
                'user': [AccessType.READ],
                'guest': []
            },
            SecurityLevel.RESTRICTED: {
                'owner': [AccessType.READ, AccessType.WRITE, AccessType.DELETE, AccessType.ADMIN],
                'admin': [AccessType.READ, AccessType.WRITE, AccessType.DELETE, AccessType.ADMIN],
                'user': [],
                'guest': []
            }
        }
    
    def _get_permission_key(self, user_id: str, file_path: str) -> str:
        """生成权限键"""
        return f"{user_id}:{file_path}"
    
    def _log_access(self, request: AccessRequest, result: PermissionResult, reason: Optional[str] = None):
        """记录访问日志"""
        log = AccessLog(
            id=f"{request.user_id}_{request.file_path}_{int(request.request_time.timestamp())}",
            user_id=request.user_id,
            file_path=request.file_path,
            access_type=request.access_type,
            result=result,
            timestamp=request.request_time,
            ip_address=request.ip_address,
            user_agent=request.user_agent,
            reason=reason,
            metadata=request.metadata
        )
        
        self.access_logs.append(log)
        self._save_access_logs()
    
    def check_file_permission(self, user_id: str, file_path: str, access_type: AccessType,
                            ip_address: Optional[str] = None, user_agent: Optional[str] = None) -> Tuple[bool, str]:
        """检查文件权限
        
        Args:
            user_id: 用户ID
            file_path: 文件路径
            access_type: 访问类型
            ip_address: IP地址
            user_agent: 用户代理
            
        Returns:
            (是否有权限, 原因)
        """
        request = AccessRequest(
            user_id=user_id,
            file_path=file_path,
            access_type=access_type,
            request_time=datetime.now(),
            ip_address=ip_address,
            user_agent=user_agent
        )
        
        try:
            # 1. 检查显式权限
            permission_key = self._get_permission_key(user_id, file_path)
            if permission_key in self.permissions:
                permission = self.permissions[permission_key]
                
                # 检查权限是否过期
                if permission.expires_at and permission.expires_at < datetime.now():
                    self._log_access(request, PermissionResult.DENIED, "权限已过期")
                    return False, "权限已过期"
                
                # 检查访问类型
                if access_type in permission.access_types:
                    # 检查条件
                    if permission.conditions:
                        if not self._check_conditions(permission.conditions, request):
                            self._log_access(request, PermissionResult.DENIED, "不满足访问条件")
                            return False, "不满足访问条件"
                    
                    self._log_access(request, PermissionResult.GRANTED, "显式权限")
                    return True, "显式权限"
                else:
                    self._log_access(request, PermissionResult.DENIED, "访问类型不匹配")
                    return False, "访问类型不匹配"
            
            # 2. 检查基于角色的权限
            has_permission, reason = self._check_role_based_permission(user_id, file_path, access_type)
            if has_permission:
                self._log_access(request, PermissionResult.GRANTED, f"角色权限: {reason}")
                return True, f"角色权限: {reason}"
            
            # 3. 检查文件所有者权限
            if self._is_file_owner(user_id, file_path):
                self._log_access(request, PermissionResult.GRANTED, "文件所有者")
                return True, "文件所有者"
            
            # 4. 检查默认权限
            has_permission, reason = self._check_default_permission(user_id, file_path, access_type)
            if has_permission:
                self._log_access(request, PermissionResult.GRANTED, f"默认权限: {reason}")
                return True, f"默认权限: {reason}"
            
            # 5. 默认拒绝
            self._log_access(request, PermissionResult.DENIED, "无权限")
            return False, "无权限"
            
        except Exception as e:
            logger.error(f"检查文件权限失败: {e}")
            self._log_access(request, PermissionResult.DENIED, f"系统错误: {e}")
            return False, f"系统错误: {e}"
    
    def _check_conditions(self, conditions: Dict[str, Any], request: AccessRequest) -> bool:
        """检查访问条件"""
        try:
            # 时间条件
            if 'time_range' in conditions:
                time_range = conditions['time_range']
                current_time = request.request_time.time()
                start_time = datetime.strptime(time_range['start'], '%H:%M').time()
                end_time = datetime.strptime(time_range['end'], '%H:%M').time()
                
                if not (start_time <= current_time <= end_time):
                    return False
            
            # IP地址条件
            if 'allowed_ips' in conditions and request.ip_address:
                allowed_ips = conditions['allowed_ips']
                if request.ip_address not in allowed_ips:
                    return False
            
            # 用户代理条件
            if 'allowed_user_agents' in conditions and request.user_agent:
                allowed_patterns = conditions['allowed_user_agents']
                import re
                if not any(re.search(pattern, request.user_agent) for pattern in allowed_patterns):
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"检查访问条件失败: {e}")
            return False
    
    def _check_role_based_permission(self, user_id: str, file_path: str, access_type: AccessType) -> Tuple[bool, str]:
        """检查基于角色的权限"""
        try:
            # 获取用户角色
            user_roles = self.access_control.get_user_roles(user_id)
            
            # 检查管理员权限
            if 'admin' in user_roles:
                return True, "管理员权限"
            
            # 检查文件特定权限
            file_permissions = self.access_control.get_user_permissions(user_id)
            for permission in file_permissions:
                if permission.get('resource') == file_path and permission.get('action') == access_type.value:
                    return True, f"角色权限: {permission.get('role')}"
            
            return False, "无角色权限"
            
        except Exception as e:
            logger.error(f"检查角色权限失败: {e}")
            return False, f"角色权限检查错误: {e}"
    
    def _is_file_owner(self, user_id: str, file_path: str) -> bool:
        """检查是否为文件所有者"""
        try:
            # 这里应该从文件元数据或数据库中获取文件所有者信息
            # 暂时使用简单的路径匹配
            if file_path.startswith(f"/users/{user_id}/"):
                return True
            
            # 检查路径中是否包含用户ID作为目录
            path_parts = file_path.split('/')
            if user_id in path_parts:
                # 确保用户ID是作为目录名出现，而不是文件名的一部分
                user_index = path_parts.index(user_id)
                if user_index > 0 and user_index < len(path_parts) - 1:
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"检查文件所有者失败: {e}")
            return False
    
    def _check_default_permission(self, user_id: str, file_path: str, access_type: AccessType) -> Tuple[bool, str]:
        """检查默认权限"""
        try:
            # 获取文件安全级别（这里需要从文件元数据获取）
            security_level = self._get_file_security_level(file_path)
            
            # 获取用户角色
            user_roles = self.access_control.get_user_roles(user_id)
            
            # 检查默认权限
            default_perms = self.default_permissions.get(security_level, {})
            
            for role in user_roles:
                if role in default_perms and access_type in default_perms[role]:
                    return True, f"默认{role}权限"
            
            # 检查guest权限
            if 'guest' in default_perms and access_type in default_perms['guest']:
                return True, "默认guest权限"
            
            return False, "无默认权限"
            
        except Exception as e:
            logger.error(f"检查默认权限失败: {e}")
            return False, f"默认权限检查错误: {e}"
    
    def _get_file_security_level(self, file_path: str) -> SecurityLevel:
        """获取文件安全级别"""
        # 这里应该从文件元数据或数据库中获取
        # 暂时使用路径匹配
        if '/public/' in file_path:
            return SecurityLevel.PUBLIC
        elif '/internal/' in file_path:
            return SecurityLevel.INTERNAL
        elif '/confidential/' in file_path:
            return SecurityLevel.CONFIDENTIAL
        elif '/restricted/' in file_path:
            return SecurityLevel.RESTRICTED
        else:
            return SecurityLevel.INTERNAL  # 默认级别
    
    def grant_file_permission(self, user_id: str, file_path: str, access_types: List[AccessType],
                            granted_by: str, expires_at: Optional[datetime] = None,
                            conditions: Optional[Dict[str, Any]] = None) -> bool:
        """授予文件权限
        
        Args:
            user_id: 用户ID
            file_path: 文件路径
            access_types: 访问类型列表
            granted_by: 授权者
            expires_at: 过期时间
            conditions: 访问条件
            
        Returns:
            是否成功
        """
        try:
            permission_key = self._get_permission_key(user_id, file_path)
            
            permission = FilePermission(
                file_path=file_path,
                user_id=user_id,
                access_types=access_types,
                granted_by=granted_by,
                granted_at=datetime.now(),
                expires_at=expires_at,
                conditions=conditions
            )
            
            self.permissions[permission_key] = permission
            self._save_permissions()
            
            logger.info(f"授予用户 {user_id} 文件 {file_path} 权限: {[at.value for at in access_types]}")
            return True
            
        except Exception as e:
            logger.error(f"授予文件权限失败: {e}")
            return False
    
    def revoke_file_permission(self, user_id: str, file_path: str) -> bool:
        """撤销文件权限
        
        Args:
            user_id: 用户ID
            file_path: 文件路径
            
        Returns:
            是否成功
        """
        try:
            permission_key = self._get_permission_key(user_id, file_path)
            
            if permission_key in self.permissions:
                del self.permissions[permission_key]
                self._save_permissions()
                
                logger.info(f"撤销用户 {user_id} 文件 {file_path} 权限")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"撤销文件权限失败: {e}")
            return False
    
    def list_user_permissions(self, user_id: str) -> List[FilePermission]:
        """列出用户权限
        
        Args:
            user_id: 用户ID
            
        Returns:
            权限列表
        """
        try:
            permissions = []
            for permission in self.permissions.values():
                if permission.user_id == user_id:
                    # 检查是否过期
                    if not permission.expires_at or permission.expires_at > datetime.now():
                        permissions.append(permission)
            
            return permissions
            
        except Exception as e:
            logger.error(f"列出用户权限失败: {e}")
            return []
    
    def list_file_permissions(self, file_path: str) -> List[FilePermission]:
        """列出文件权限
        
        Args:
            file_path: 文件路径
            
        Returns:
            权限列表
        """
        try:
            permissions = []
            for permission in self.permissions.values():
                if permission.file_path == file_path:
                    # 检查是否过期
                    if not permission.expires_at or permission.expires_at > datetime.now():
                        permissions.append(permission)
            
            return permissions
            
        except Exception as e:
            logger.error(f"列出文件权限失败: {e}")
            return []
    
    def get_access_logs(self, user_id: Optional[str] = None, file_path: Optional[str] = None,
                       start_time: Optional[datetime] = None, end_time: Optional[datetime] = None,
                       limit: int = 100) -> List[AccessLog]:
        """获取访问日志
        
        Args:
            user_id: 用户ID过滤
            file_path: 文件路径过滤
            start_time: 开始时间
            end_time: 结束时间
            limit: 限制数量
            
        Returns:
            访问日志列表
        """
        try:
            logs = self.access_logs
            
            # 过滤条件
            if user_id:
                logs = [log for log in logs if log.user_id == user_id]
            
            if file_path:
                logs = [log for log in logs if log.file_path == file_path]
            
            if start_time:
                logs = [log for log in logs if log.timestamp >= start_time]
            
            if end_time:
                logs = [log for log in logs if log.timestamp <= end_time]
            
            # 按时间倒序排列
            logs.sort(key=lambda x: x.timestamp, reverse=True)
            
            # 限制数量
            return logs[:limit]
            
        except Exception as e:
            logger.error(f"获取访问日志失败: {e}")
            return []
    
    def cleanup_expired_permissions(self) -> int:
        """清理过期权限
        
        Returns:
            清理的权限数量
        """
        try:
            current_time = datetime.now()
            expired_keys = []
            
            for key, permission in self.permissions.items():
                if permission.expires_at and permission.expires_at < current_time:
                    expired_keys.append(key)
            
            for key in expired_keys:
                del self.permissions[key]
            
            if expired_keys:
                self._save_permissions()
                logger.info(f"清理了 {len(expired_keys)} 个过期权限")
            
            return len(expired_keys)
            
        except Exception as e:
            logger.error(f"清理过期权限失败: {e}")
            return 0
    
    def get_permission_statistics(self) -> Dict[str, Any]:
        """获取权限统计信息
        
        Returns:
            统计信息
        """
        try:
            current_time = datetime.now()
            
            # 权限统计
            total_permissions = len(self.permissions)
            active_permissions = sum(1 for p in self.permissions.values() 
                                   if not p.expires_at or p.expires_at > current_time)
            expired_permissions = total_permissions - active_permissions
            
            # 按访问类型统计
            access_type_stats = {}
            for permission in self.permissions.values():
                for access_type in permission.access_types:
                    access_type_stats[access_type.value] = access_type_stats.get(access_type.value, 0) + 1
            
            # 访问日志统计
            recent_logs = [log for log in self.access_logs 
                          if log.timestamp > current_time - timedelta(days=7)]
            
            access_stats = {
                'granted': sum(1 for log in recent_logs if log.result == PermissionResult.GRANTED),
                'denied': sum(1 for log in recent_logs if log.result == PermissionResult.DENIED),
                'conditional': sum(1 for log in recent_logs if log.result == PermissionResult.CONDITIONAL)
            }
            
            return {
                'permissions': {
                    'total': total_permissions,
                    'active': active_permissions,
                    'expired': expired_permissions,
                    'by_access_type': access_type_stats
                },
                'access_logs': {
                    'total': len(self.access_logs),
                    'recent_week': len(recent_logs),
                    'by_result': access_stats
                }
            }
            
        except Exception as e:
            logger.error(f"获取权限统计失败: {e}")
            return {}
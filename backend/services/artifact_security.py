#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""安全工件与文件策略服务

提供文件和工件的安全管理、验证、访问控制和审计功能。
使用仓库层进行数据持久化。
"""

import hashlib
import logging
import mimetypes
import os
import shutil
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple

logger = logging.getLogger(__name__)


# ==================== 枚举类型 ====================

class SecurityLevel(Enum):
    """安全级别"""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"


class FileType(Enum):
    """文件类型"""
    DOCUMENT = "document"
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
    ARCHIVE = "archive"
    CODE = "code"
    DATA = "data"
    MODEL = "model"
    UNKNOWN = "unknown"


class ArtifactType(Enum):
    """工件类型"""
    TRAINING_DATA = "training_data"
    MODEL_FILE = "model_file"
    CONFIG_FILE = "config_file"
    LOG_FILE = "log_file"
    REPORT = "report"
    BACKUP = "backup"
    TEMP = "temp"


class ArtifactStatus(Enum):
    """工件状态"""
    ACTIVE = "active"
    ARCHIVED = "archived"
    DELETED = "deleted"
    PENDING = "pending"
    LOCKED = "locked"


# ==================== 数据类 ====================

@dataclass
class SecurityPolicy:
    """安全策略"""
    id: str
    name: str
    description: str
    security_level: SecurityLevel
    allowed_file_types: List[str]
    max_file_size: int  # bytes
    encryption_required: bool
    virus_scan_required: bool
    access_control_enabled: bool
    audit_enabled: bool
    retention_days: int
    created_at: datetime
    updated_at: datetime
    is_default: bool = False
    is_active: bool = True
    tenant_id: str = None
    created_by: str = None


@dataclass
class FileMetadata:
    """文件元数据"""
    id: str
    original_name: str
    stored_name: str
    file_path: str
    file_type: FileType
    mime_type: str
    size: int
    hash_sha256: str
    security_level: SecurityLevel
    artifact_type: ArtifactType
    owner_id: str
    created_at: datetime
    updated_at: datetime
    accessed_at: datetime
    tags: List[str]
    metadata: Dict[str, Any]
    tenant_id: str = None
    artifact_id: str = None
    version_id: str = None
    is_encrypted: bool = False
    encryption_key_id: str = None


@dataclass
class Artifact:
    """工件"""
    id: str
    name: str
    description: str
    artifact_type: ArtifactType
    security_level: SecurityLevel
    status: ArtifactStatus
    owner_id: str
    current_version: str
    versions: List['ArtifactVersion']
    tags: List[str]
    metadata: Dict[str, Any]
    created_at: datetime
    updated_at: datetime
    tenant_id: str = None
    version_count: int = 0
    total_size: int = 0
    policy_id: str = None


@dataclass
class ArtifactVersion:
    """工件版本"""
    id: str
    artifact_id: str
    version: str
    status: str
    file_path: str
    file_size: int
    file_hash: str
    mime_type: str
    changelog: str
    tags: List[str]
    metadata: Dict[str, Any]
    created_by: str
    created_at: datetime


@dataclass
class SecurityScanResult:
    """安全扫描结果"""
    file_id: str
    scan_type: str
    status: str  # passed, failed, warning
    threats_found: List[str]
    scan_time: datetime
    scanner_version: str
    details: Dict[str, Any]


@dataclass
class ArtifactDependency:
    """工件依赖"""
    source_artifact_id: str
    target_artifact_id: str
    dependency_type: str
    version_constraint: str


# ==================== 服务类 ====================

class ArtifactSecurityService:
    """工件安全服务
    
    提供文件和工件的安全管理功能，包括：
    - 文件上传验证
    - 安全存储
    - 访问控制
    - 安全扫描
    - 审计日志
    """
    
    def __init__(self, config: Dict[str, Any] = None, use_memory_storage: bool = False):
        self.config = config or {}
        self._use_memory_storage = use_memory_storage
        
        # 初始化仓库
        self._init_repositories()
        
        # 初始化访问控制
        self._init_access_control()
        
        # 文件类型映射
        self.file_type_mapping = self._init_file_type_mapping()
        
        # 初始化默认策略
        self._init_default_policies()
        
        # 存储路径
        self.storage_base_path = config.get('storage_path', '/var/lib/vectorsphere/artifacts') if config else '/var/lib/vectorsphere/artifacts'
        os.makedirs(self.storage_base_path, exist_ok=True)
    
    def _init_repositories(self):
        """初始化仓库"""
        try:
            from backend.repositories.artifact_repository import (
                get_security_policy_repository,
                get_artifact_repository,
                get_artifact_version_repository,
                get_file_metadata_repository,
                get_artifact_dependency_repository,
                get_artifact_access_log_repository
            )
            self._policy_repo = get_security_policy_repository(use_memory=self._use_memory_storage)
            self._artifact_repo = get_artifact_repository(use_memory=self._use_memory_storage)
            self._version_repo = get_artifact_version_repository(use_memory=self._use_memory_storage)
            self._file_repo = get_file_metadata_repository(use_memory=self._use_memory_storage)
            self._dependency_repo = get_artifact_dependency_repository(use_memory=self._use_memory_storage)
            self._access_log_repo = get_artifact_access_log_repository(use_memory=self._use_memory_storage)
        except ImportError as e:
            logger.warning(f"Failed to import repositories: {e}")
            self._policy_repo = None
            self._artifact_repo = None
            self._version_repo = None
            self._file_repo = None
            self._dependency_repo = None
            self._access_log_repo = None
    
    def _init_access_control(self):
        """初始化访问控制"""
        try:
            from backend.services.access_control import AccessControlManager
            self.access_control = AccessControlManager(self.config.get('access_control', {}))
        except ImportError:
            logger.warning("AccessControlManager not available")
            self.access_control = None
    
    # ==================== 安全策略管理 ====================
    
    def create_security_policy(self, policy: SecurityPolicy) -> bool:
        """创建安全策略
        
        Args:
            policy: 安全策略对象
            
        Returns:
            是否成功
        """
        try:
            if self._policy_repo:
                policy_data = {
                    'id': policy.id,
                    'tenant_id': policy.tenant_id,
                    'name': policy.name,
                    'description': policy.description,
                    'security_level': policy.security_level.value if isinstance(policy.security_level, SecurityLevel) else policy.security_level,
                    'allowed_file_types': policy.allowed_file_types,
                    'max_file_size': policy.max_file_size,
                    'encryption_required': policy.encryption_required,
                    'virus_scan_required': policy.virus_scan_required,
                    'access_control_enabled': policy.access_control_enabled,
                    'audit_enabled': policy.audit_enabled,
                    'retention_days': policy.retention_days,
                    'is_default': policy.is_default,
                    'is_active': policy.is_active,
                    'created_by': policy.created_by
                }
                result = self._policy_repo.create(policy_data)
                return result is not None
            return False
        except Exception as e:
            logger.error(f"Failed to create security policy: {e}")
            return False
    
    def update_security_policy(self, policy_id: str, updates: Dict[str, Any], tenant_id: str = None) -> bool:
        """更新安全策略
        
        Args:
            policy_id: 策略ID
            updates: 更新内容
            tenant_id: 租户ID
            
        Returns:
            是否成功
        """
        try:
            if self._policy_repo:
                # 转换枚举类型
                if 'security_level' in updates and isinstance(updates['security_level'], SecurityLevel):
                    updates['security_level'] = updates['security_level'].value
                return self._policy_repo.update(policy_id, updates, tenant_id)
            return False
        except Exception as e:
            logger.error(f"Failed to update security policy: {e}")
            return False
    
    def get_security_policies(self, tenant_id: str = None, security_level: str = None) -> List[SecurityPolicy]:
        """获取所有安全策略
        
        Args:
            tenant_id: 租户ID
            security_level: 安全级别
            
        Returns:
            策略列表
        """
        try:
            if self._policy_repo:
                policies_data = self._policy_repo.get_all(tenant_id, security_level)
                return [self._dict_to_policy(p) for p in policies_data]
            return []
        except Exception as e:
            logger.error(f"Failed to get security policies: {e}")
            return []
    
    def get_security_policy(self, policy_id: str, tenant_id: str = None) -> Optional[SecurityPolicy]:
        """获取单个安全策略
        
        Args:
            policy_id: 策略ID
            tenant_id: 租户ID
            
        Returns:
            安全策略
        """
        try:
            if self._policy_repo:
                policy_data = self._policy_repo.get_by_id(policy_id, tenant_id)
                if policy_data:
                    return self._dict_to_policy(policy_data)
            return None
        except Exception as e:
            logger.error(f"Failed to get security policy: {e}")
            return None
    
    def delete_security_policy(self, policy_id: str, tenant_id: str = None) -> bool:
        """删除安全策略
        
        Args:
            policy_id: 策略ID
            tenant_id: 租户ID
            
        Returns:
            是否成功
        """
        try:
            if self._policy_repo:
                return self._policy_repo.delete(policy_id, tenant_id)
            return False
        except Exception as e:
            logger.error(f"Failed to delete security policy: {e}")
            return False
    
    # ==================== 文件验证和存储 ====================
    
    def validate_file_upload(
        self,
        file_path: str,
        user_id: str,
        artifact_type: ArtifactType,
        security_level: SecurityLevel = SecurityLevel.INTERNAL,
        tenant_id: str = None
    ) -> Tuple[bool, str, Optional[FileMetadata]]:
        """验证文件上传
        
        Args:
            file_path: 文件路径
            user_id: 用户ID
            artifact_type: 工件类型
            security_level: 安全级别
            tenant_id: 租户ID
            
        Returns:
            (是否通过, 错误信息, 文件元数据)
        """
        try:
            # 1. 检查文件是否存在
            if not os.path.exists(file_path):
                return False, "File does not exist", None
            
            # 2. 获取文件信息
            file_size = os.path.getsize(file_path)
            file_hash = self._calculate_file_hash(file_path)
            
            # 3. 检测文件类型
            file_type = self._detect_file_type(file_path)
            mime_type = mimetypes.guess_type(file_path)[0] or 'application/octet-stream'
            
            # 4. 获取适用的安全策略
            policy = self._get_applicable_policy(artifact_type, security_level, tenant_id)
            if not policy:
                return False, "No applicable security policy found", None
            
            # 5. 验证文件类型
            if not self._validate_file_type(file_path, policy):
                allowed = ', '.join(policy.allowed_file_types)
                return False, f"Unsupported file type. Allowed types: {allowed}", None
            
            # 6. 验证文件大小
            if file_size > policy.max_file_size:
                return False, f"File size exceeds limit ({policy.max_file_size} bytes)", None
            
            # 7. 检查访问权限
            if self.access_control and policy.access_control_enabled:
                access_result = self.access_control.check_permission(
                    user_id, f"artifact:{artifact_type.value}", "upload"
                )
                if not access_result.allowed:
                    return False, f"Permission denied: {access_result.reason}", None
            
            # 8. 安全扫描
            if policy.virus_scan_required:
                scan_result = self._perform_security_scan(file_path)
                if scan_result.status == "failed":
                    threats = ', '.join(scan_result.threats_found)
                    return False, f"Security scan failed: {threats}", None
            
            # 9. 创建文件元数据
            file_metadata = FileMetadata(
                id=str(uuid.uuid4()),
                original_name=os.path.basename(file_path),
                stored_name=f"{uuid.uuid4().hex}_{os.path.basename(file_path)}",
                file_path=file_path,
                file_type=file_type,
                mime_type=mime_type,
                size=file_size,
                hash_sha256=file_hash,
                security_level=security_level,
                artifact_type=artifact_type,
                owner_id=user_id,
                created_at=datetime.now(),
                updated_at=datetime.now(),
                accessed_at=datetime.now(),
                tags=[],
                metadata={},
                tenant_id=tenant_id
            )
            
            # 10. 存储元数据
            if self._file_repo:
                file_data = self._file_metadata_to_dict(file_metadata)
                self._file_repo.create(file_data)
            
            # 11. 记录审计日志
            self._log_operation("validate", "file", file_metadata.id, user_id, "success", tenant_id)
            
            return True, "Validation passed", file_metadata
            
        except Exception as e:
            logger.error(f"File upload validation failed: {e}")
            return False, f"Validation failed: {str(e)}", None
    
    def store_secure_file(
        self,
        file_metadata: FileMetadata,
        source_path: str,
        tenant_id: str = None
    ) -> Tuple[bool, str]:
        """安全存储文件
        
        Args:
            file_metadata: 文件元数据
            source_path: 源文件路径
            tenant_id: 租户ID
            
        Returns:
            (是否成功, 存储路径或错误信息)
        """
        try:
            # 1. 创建存储目录结构
            storage_dir = os.path.join(
                self.storage_base_path,
                file_metadata.security_level.value if isinstance(file_metadata.security_level, SecurityLevel) else file_metadata.security_level,
                file_metadata.artifact_type.value if isinstance(file_metadata.artifact_type, ArtifactType) else file_metadata.artifact_type,
                file_metadata.created_at.strftime('%Y/%m/%d')
            )
            os.makedirs(storage_dir, exist_ok=True)
            
            # 2. 构建存储路径
            storage_path = os.path.join(storage_dir, file_metadata.stored_name)
            
            # 3. 获取安全策略
            artifact_type = file_metadata.artifact_type if isinstance(file_metadata.artifact_type, ArtifactType) else ArtifactType(file_metadata.artifact_type)
            security_level = file_metadata.security_level if isinstance(file_metadata.security_level, SecurityLevel) else SecurityLevel(file_metadata.security_level)
            policy = self._get_applicable_policy(artifact_type, security_level, tenant_id)
            
            # 4. 复制文件
            shutil.copy2(source_path, storage_path)
            
            # 5. 加密文件（如果需要）
            if policy and policy.encryption_required:
                encrypted_path = self._encrypt_file(storage_path)
                if encrypted_path:
                    os.remove(storage_path)
                    storage_path = encrypted_path
                    file_metadata.is_encrypted = True
            
            # 6. 设置文件权限
            self._set_file_permissions(storage_path, security_level)
            
            # 7. 更新文件元数据
            file_metadata.file_path = storage_path
            file_metadata.updated_at = datetime.now()
            
            if self._file_repo:
                self._file_repo.update(file_metadata.id, {
                    'file_path': storage_path,
                    'is_encrypted': file_metadata.is_encrypted
                }, tenant_id)
            
            # 8. 记录审计日志
            self._log_operation("store", "file", file_metadata.id, file_metadata.owner_id, "success", tenant_id)
            
            return True, storage_path
            
        except Exception as e:
            logger.error(f"File storage failed: {e}")
            return False, f"Storage failed: {str(e)}"
    
    def access_file(
        self,
        file_id: str,
        user_id: str,
        operation: str = "read",
        tenant_id: str = None
    ) -> Tuple[bool, str, Optional[str]]:
        """访问文件
        
        Args:
            file_id: 文件ID
            user_id: 用户ID
            operation: 操作类型
            tenant_id: 租户ID
            
        Returns:
            (是否允许, 错误信息, 文件路径)
        """
        try:
            # 1. 获取文件元数据
            if not self._file_repo:
                return False, "Repository not available", None
            
            file_data = self._file_repo.get_by_id(file_id, tenant_id)
            if not file_data:
                return False, "File not found", None
            
            # 2. 检查访问权限
            if self.access_control:
                access_result = self.access_control.check_permission(
                    user_id, f"file:{file_id}", operation
                )
                if not access_result.allowed:
                    self._log_operation(operation, "file", file_id, user_id, "denied", tenant_id)
                    return False, f"Permission denied: {access_result.reason}", None
            
            # 3. 检查文件是否存在
            file_path = file_data.get('file_path')
            if not file_path or not os.path.exists(file_path):
                return False, "File is damaged or missing", None
            
            # 4. 更新访问时间
            self._file_repo.update_access_time(file_id)
            
            # 5. 记录访问日志
            self._log_operation(operation, "file", file_id, user_id, "success", tenant_id)
            
            return True, "Access allowed", file_path
            
        except Exception as e:
            logger.error(f"File access failed: {e}")
            return False, f"Access failed: {str(e)}", None
    
    def delete_file(
        self,
        file_id: str,
        user_id: str,
        tenant_id: str = None
    ) -> Tuple[bool, str]:
        """删除文件
        
        Args:
            file_id: 文件ID
            user_id: 用户ID
            tenant_id: 租户ID
            
        Returns:
            (是否成功, 错误信息)
        """
        try:
            # 1. 获取文件元数据
            if not self._file_repo:
                return False, "Repository not available"
            
            file_data = self._file_repo.get_by_id(file_id, tenant_id)
            if not file_data:
                return False, "File not found"
            
            # 2. 检查删除权限
            if self.access_control:
                access_result = self.access_control.check_permission(
                    user_id, f"file:{file_id}", "delete"
                )
                if not access_result.allowed:
                    return False, f"Permission denied: {access_result.reason}"
            
            # 3. 删除物理文件
            file_path = file_data.get('file_path')
            if file_path and os.path.exists(file_path):
                os.remove(file_path)
            
            # 4. 删除元数据
            self._file_repo.delete(file_id, tenant_id)
            
            # 5. 记录删除日志
            self._log_operation("delete", "file", file_id, user_id, "success", tenant_id)
            
            return True, "Delete successful"
            
        except Exception as e:
            logger.error(f"File deletion failed: {e}")
            return False, f"Deletion failed: {str(e)}"
    
    def list_files(
        self,
        user_id: str,
        tenant_id: str = None,
        artifact_type: str = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict]:
        """列出文件
        
        Args:
            user_id: 用户ID
            tenant_id: 租户ID
            artifact_type: 工件类型
            limit: 限制数量
            offset: 偏移量
            
        Returns:
            文件列表
        """
        try:
            if not self._file_repo:
                return []
            
            return self._file_repo.get_by_owner(
                owner_id=user_id,
                tenant_id=tenant_id,
                artifact_type=artifact_type,
                limit=limit,
                offset=offset
            )
            
        except Exception as e:
            logger.error(f"Failed to list files: {e}")
            return []
    
    def get_file_metadata(
        self,
        file_id: str,
        user_id: str,
        tenant_id: str = None
    ) -> Optional[Dict]:
        """获取文件元数据
        
        Args:
            file_id: 文件ID
            user_id: 用户ID
            tenant_id: 租户ID
            
        Returns:
            文件元数据
        """
        try:
            if not self._file_repo:
                return None
            
            file_data = self._file_repo.get_by_id(file_id, tenant_id)
            if not file_data:
                return None
            
            # 检查访问权限
            if self.access_control:
                access_result = self.access_control.check_permission(
                    user_id, f"file:{file_id}", "read"
                )
                if not access_result.allowed:
                    return None
            
            return file_data
            
        except Exception as e:
            logger.error(f"Failed to get file metadata: {e}")
            return None
    
    # ==================== 工件管理 ====================
    
    def create_artifact(
        self,
        name: str,
        description: str,
        artifact_type: ArtifactType,
        security_level: SecurityLevel,
        owner_id: str,
        tags: List[str] = None,
        metadata: Dict[str, Any] = None,
        tenant_id: str = None,
        policy_id: str = None
    ) -> Tuple[bool, str, Optional[Artifact]]:
        """创建工件
        
        Args:
            name: 工件名称
            description: 描述
            artifact_type: 工件类型
            security_level: 安全级别
            owner_id: 所有者ID
            tags: 标签
            metadata: 元数据
            tenant_id: 租户ID
            policy_id: 策略ID
            
        Returns:
            (是否成功, 错误信息, 工件对象)
        """
        try:
            if not self._artifact_repo:
                return False, "Repository not available", None
            
            artifact_data = {
                'tenant_id': tenant_id,
                'name': name,
                'description': description,
                'artifact_type': artifact_type.value if isinstance(artifact_type, ArtifactType) else artifact_type,
                'security_level': security_level.value if isinstance(security_level, SecurityLevel) else security_level,
                'status': 'active',
                'owner_id': owner_id,
                'tags': tags or [],
                'metadata': metadata or {},
                'policy_id': policy_id
            }
            
            result = self._artifact_repo.create(artifact_data)
            if result:
                artifact = self._dict_to_artifact(result)
                self._log_operation("create", "artifact", artifact.id, owner_id, "success", tenant_id)
                return True, "Artifact created successfully", artifact
            
            return False, "Failed to create artifact", None
            
        except Exception as e:
            logger.error(f"Failed to create artifact: {e}")
            return False, f"Creation failed: {str(e)}", None
    
    def get_artifact(
        self,
        artifact_id: str,
        user_id: str,
        tenant_id: str = None
    ) -> Optional[Artifact]:
        """获取工件
        
        Args:
            artifact_id: 工件ID
            user_id: 用户ID
            tenant_id: 租户ID
            
        Returns:
            工件对象
        """
        try:
            if not self._artifact_repo:
                return None
            
            artifact_data = self._artifact_repo.get_by_id(artifact_id, tenant_id)
            if not artifact_data:
                return None
            
            # 检查访问权限
            if self.access_control:
                access_result = self.access_control.check_permission(
                    user_id, f"artifact:{artifact_id}", "read"
                )
                if not access_result.allowed:
                    return None
            
            return self._dict_to_artifact(artifact_data)
            
        except Exception as e:
            logger.error(f"Failed to get artifact: {e}")
            return None
    
    def list_artifacts(
        self,
        user_id: str,
        filters: Dict[str, Any] = None,
        tenant_id: str = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Artifact]:
        """列出工件
        
        Args:
            user_id: 用户ID
            filters: 过滤条件
            tenant_id: 租户ID
            limit: 限制数量
            offset: 偏移量
            
        Returns:
            工件列表
        """
        try:
            if not self._artifact_repo:
                return []
            
            filters = filters or {}
            artifacts_data = self._artifact_repo.get_all(
                tenant_id=tenant_id,
                owner_id=filters.get('owner_id'),
                artifact_type=filters.get('artifact_type'),
                security_level=filters.get('security_level'),
                status=filters.get('status'),
                limit=limit,
                offset=offset
            )
            
            result = []
            for a in artifacts_data:
                # 检查访问权限
                if self.access_control:
                    access_result = self.access_control.check_permission(
                        user_id, f"artifact:{a['id']}", "read"
                    )
                    if not access_result.allowed:
                        continue
                result.append(self._dict_to_artifact(a))
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to list artifacts: {e}")
            return []
    
    def update_artifact_status(
        self,
        artifact_id: str,
        status: ArtifactStatus,
        user_id: str,
        tenant_id: str = None
    ) -> Tuple[bool, str]:
        """更新工件状态
        
        Args:
            artifact_id: 工件ID
            status: 新状态
            user_id: 用户ID
            tenant_id: 租户ID
            
        Returns:
            (是否成功, 错误信息)
        """
        try:
            if not self._artifact_repo:
                return False, "Repository not available"
            
            # 检查权限
            if self.access_control:
                access_result = self.access_control.check_permission(
                    user_id, f"artifact:{artifact_id}", "update"
                )
                if not access_result.allowed:
                    return False, f"Permission denied: {access_result.reason}"
            
            status_value = status.value if isinstance(status, ArtifactStatus) else status
            success = self._artifact_repo.update_status(artifact_id, status_value, tenant_id)
            
            if success:
                self._log_operation("update_status", "artifact", artifact_id, user_id, "success", tenant_id)
                return True, "Status updated successfully"
            
            return False, "Artifact not found or update failed"
            
        except Exception as e:
            logger.error(f"Failed to update artifact status: {e}")
            return False, f"Update failed: {str(e)}"
    
    # ==================== 版本管理 ====================
    
    def upload_artifact_version(
        self,
        artifact_id: str,
        file_path: str,
        version: str,
        changelog: str,
        user_id: str,
        tags: List[str] = None,
        tenant_id: str = None
    ) -> Tuple[bool, str, Optional[ArtifactVersion]]:
        """上传工件版本
        
        Args:
            artifact_id: 工件ID
            file_path: 文件路径
            version: 版本号
            changelog: 变更日志
            user_id: 用户ID
            tags: 标签
            tenant_id: 租户ID
            
        Returns:
            (是否成功, 错误信息, 版本对象)
        """
        try:
            if not self._artifact_repo or not self._version_repo:
                return False, "Repository not available", None
            
            # 1. 获取工件
            artifact_data = self._artifact_repo.get_by_id(artifact_id, tenant_id)
            if not artifact_data:
                return False, "Artifact not found", None
            
            # 2. 检查权限
            if self.access_control:
                access_result = self.access_control.check_permission(
                    user_id, f"artifact:{artifact_id}", "upload"
                )
                if not access_result.allowed:
                    return False, f"Permission denied: {access_result.reason}", None
            
            # 3. 验证文件
            artifact_type = ArtifactType(artifact_data['artifact_type'])
            security_level = SecurityLevel(artifact_data['security_level'])
            
            valid, msg, file_metadata = self.validate_file_upload(
                file_path, user_id, artifact_type, security_level, tenant_id
            )
            if not valid:
                return False, msg, None
            
            # 4. 存储文件
            success, storage_path = self.store_secure_file(file_metadata, file_path, tenant_id)
            if not success:
                return False, storage_path, None
            
            # 5. 创建版本记录
            version_data = {
                'artifact_id': artifact_id,
                'version': version,
                'status': 'active',
                'file_path': storage_path,
                'file_size': file_metadata.size,
                'file_hash': file_metadata.hash_sha256,
                'mime_type': file_metadata.mime_type,
                'changelog': changelog,
                'tags': tags or [],
                'metadata': {},
                'created_by': user_id
            }
            
            version_result = self._version_repo.create(version_data)
            if not version_result:
                return False, "Failed to create version record", None
            
            # 6. 更新工件
            self._artifact_repo.update(artifact_id, {'current_version': version}, tenant_id)
            self._artifact_repo.increment_version_count(artifact_id, file_metadata.size)
            
            # 7. 更新文件元数据关联
            if self._file_repo:
                self._file_repo.update(file_metadata.id, {
                    'artifact_id': artifact_id,
                    'version_id': version_result['id']
                }, tenant_id)
            
            artifact_version = self._dict_to_version(version_result)
            self._log_operation("upload_version", "artifact", artifact_id, user_id, "success", tenant_id)
            
            return True, "Version uploaded successfully", artifact_version
            
        except Exception as e:
            logger.error(f"Failed to upload artifact version: {e}")
            return False, f"Upload failed: {str(e)}", None
    
    def get_artifact_versions(
        self,
        artifact_id: str,
        user_id: str,
        tenant_id: str = None
    ) -> List[ArtifactVersion]:
        """获取工件的所有版本
        
        Args:
            artifact_id: 工件ID
            user_id: 用户ID
            tenant_id: 租户ID
            
        Returns:
            版本列表
        """
        try:
            if not self._version_repo:
                return []
            
            # 检查权限
            if self.access_control:
                access_result = self.access_control.check_permission(
                    user_id, f"artifact:{artifact_id}", "read"
                )
                if not access_result.allowed:
                    return []
            
            versions_data = self._version_repo.get_by_artifact(artifact_id)
            return [self._dict_to_version(v) for v in versions_data]
            
        except Exception as e:
            logger.error(f"Failed to get artifact versions: {e}")
            return []
    
    def download_artifact_version(
        self,
        artifact_id: str,
        version: str,
        user_id: str,
        tenant_id: str = None
    ) -> Tuple[bool, str, Optional[str]]:
        """下载工件版本
        
        Args:
            artifact_id: 工件ID
            version: 版本号
            user_id: 用户ID
            tenant_id: 租户ID
            
        Returns:
            (是否成功, 错误信息, 文件路径)
        """
        try:
            if not self._version_repo:
                return False, "Repository not available", None
            
            # 1. 检查权限
            if self.access_control:
                access_result = self.access_control.check_permission(
                    user_id, f"artifact:{artifact_id}", "download"
                )
                if not access_result.allowed:
                    return False, f"Permission denied: {access_result.reason}", None
            
            # 2. 获取版本
            version_data = self._version_repo.get_by_artifact_and_version(artifact_id, version)
            if not version_data:
                return False, "Version not found", None
            
            # 3. 检查文件
            file_path = version_data.get('file_path')
            if not file_path or not os.path.exists(file_path):
                return False, "File not found or damaged", None
            
            # 4. 记录访问日志
            self._log_operation("download", "artifact", artifact_id, user_id, "success", tenant_id)
            
            return True, "Download ready", file_path
            
        except Exception as e:
            logger.error(f"Failed to download artifact version: {e}")
            return False, f"Download failed: {str(e)}", None
    
    def delete_artifact_version(
        self,
        artifact_id: str,
        version: str,
        user_id: str,
        tenant_id: str = None
    ) -> Tuple[bool, str]:
        """删除工件版本
        
        Args:
            artifact_id: 工件ID
            version: 版本号
            user_id: 用户ID
            tenant_id: 租户ID
            
        Returns:
            (是否成功, 错误信息)
        """
        try:
            if not self._version_repo:
                return False, "Repository not available"
            
            # 1. 检查权限
            if self.access_control:
                access_result = self.access_control.check_permission(
                    user_id, f"artifact:{artifact_id}", "delete"
                )
                if not access_result.allowed:
                    return False, f"Permission denied: {access_result.reason}"
            
            # 2. 获取版本
            version_data = self._version_repo.get_by_artifact_and_version(artifact_id, version)
            if not version_data:
                return False, "Version not found"
            
            # 3. 删除文件
            file_path = version_data.get('file_path')
            if file_path and os.path.exists(file_path):
                os.remove(file_path)
            
            # 4. 删除版本记录
            self._version_repo.delete(version_data['id'])
            
            # 5. 记录日志
            self._log_operation("delete_version", "artifact", artifact_id, user_id, "success", tenant_id)
            
            return True, "Version deleted successfully"
            
        except Exception as e:
            logger.error(f"Failed to delete artifact version: {e}")
            return False, f"Deletion failed: {str(e)}"
    
    # ==================== 依赖管理 ====================
    
    def add_dependency(
        self,
        source_id: str,
        target_id: str,
        dependency_type: str,
        version_constraint: str,
        user_id: str,
        tenant_id: str = None
    ) -> Tuple[bool, str]:
        """添加依赖
        
        Args:
            source_id: 源工件ID
            target_id: 目标工件ID
            dependency_type: 依赖类型
            version_constraint: 版本约束
            user_id: 用户ID
            tenant_id: 租户ID
            
        Returns:
            (是否成功, 错误信息)
        """
        try:
            if not self._dependency_repo:
                return False, "Repository not available"
            
            # 检查权限
            if self.access_control:
                access_result = self.access_control.check_permission(
                    user_id, f"artifact:{source_id}", "update"
                )
                if not access_result.allowed:
                    return False, f"Permission denied: {access_result.reason}"
            
            dep_data = {
                'source_artifact_id': source_id,
                'target_artifact_id': target_id,
                'dependency_type': dependency_type,
                'version_constraint': version_constraint,
                'created_by': user_id
            }
            
            result = self._dependency_repo.create(dep_data)
            if result:
                return True, "Dependency added successfully"
            
            return False, "Failed to add dependency"
            
        except Exception as e:
            logger.error(f"Failed to add dependency: {e}")
            return False, f"Failed: {str(e)}"
    
    def remove_dependency(
        self,
        source_id: str,
        target_id: str,
        user_id: str,
        tenant_id: str = None
    ) -> Tuple[bool, str]:
        """移除依赖
        
        Args:
            source_id: 源工件ID
            target_id: 目标工件ID
            user_id: 用户ID
            tenant_id: 租户ID
            
        Returns:
            (是否成功, 错误信息)
        """
        try:
            if not self._dependency_repo:
                return False, "Repository not available"
            
            # 检查权限
            if self.access_control:
                access_result = self.access_control.check_permission(
                    user_id, f"artifact:{source_id}", "update"
                )
                if not access_result.allowed:
                    return False, f"Permission denied: {access_result.reason}"
            
            success = self._dependency_repo.delete(source_id, target_id)
            if success:
                return True, "Dependency removed successfully"
            
            return False, "Dependency not found"
            
        except Exception as e:
            logger.error(f"Failed to remove dependency: {e}")
            return False, f"Failed: {str(e)}"
    
    def get_artifact_dependencies(self, artifact_id: str) -> List[ArtifactDependency]:
        """获取工件的依赖列表"""
        try:
            if not self._dependency_repo:
                return []
            
            deps_data = self._dependency_repo.get_dependencies(artifact_id)
            return [
                ArtifactDependency(
                    source_artifact_id=d['source_artifact_id'],
                    target_artifact_id=d['target_artifact_id'],
                    dependency_type=d['dependency_type'],
                    version_constraint=d['version_constraint']
                )
                for d in deps_data
            ]
            
        except Exception as e:
            logger.error(f"Failed to get dependencies: {e}")
            return []
    
    def get_artifact_dependents(self, artifact_id: str) -> List[ArtifactDependency]:
        """获取依赖此工件的列表"""
        try:
            if not self._dependency_repo:
                return []
            
            deps_data = self._dependency_repo.get_dependents(artifact_id)
            return [
                ArtifactDependency(
                    source_artifact_id=d['source_artifact_id'],
                    target_artifact_id=d['target_artifact_id'],
                    dependency_type=d['dependency_type'],
                    version_constraint=d['version_constraint']
                )
                for d in deps_data
            ]
            
        except Exception as e:
            logger.error(f"Failed to get dependents: {e}")
            return []
    
    # ==================== 清理功能 ====================
    
    def cleanup_expired_files(self, tenant_id: str = None) -> int:
        """清理过期文件
        
        Returns:
            清理的文件数量
        """
        try:
            if not self._file_repo:
                return 0
            
            expired_files = self._file_repo.get_expired_files()
            cleaned_count = 0
            
            for file_data in expired_files:
                if tenant_id and file_data.get('tenant_id') != tenant_id:
                    continue
                
                file_path = file_data.get('file_path')
                if file_path and os.path.exists(file_path):
                    os.remove(file_path)
                
                self._file_repo.delete(file_data['id'], tenant_id)
                cleaned_count += 1
                
                self._log_operation("cleanup", "file", file_data['id'], "system", "success", tenant_id)
            
            return cleaned_count
            
        except Exception as e:
            logger.error(f"Failed to cleanup expired files: {e}")
            return 0
    
    def cleanup_old_versions(self, retention_days: int = 90, tenant_id: str = None) -> int:
        """清理旧版本
        
        Args:
            retention_days: 保留天数
            tenant_id: 租户ID
            
        Returns:
            清理的版本数量
        """
        try:
            if not self._artifact_repo or not self._version_repo:
                return 0
            
            cleaned_count = 0
            cutoff_date = datetime.utcnow() - timedelta(days=retention_days)
            
            # 获取所有工件
            artifacts = self._artifact_repo.get_all(tenant_id=tenant_id)
            
            for artifact in artifacts:
                # 获取所有版本
                versions = self._version_repo.get_by_artifact(artifact['id'])
                
                # 保留最新的3个版本，删除过期的旧版本
                if len(versions) <= 3:
                    continue
                
                for version in versions[3:]:
                    if version.get('created_at', datetime.max) < cutoff_date:
                        file_path = version.get('file_path')
                        if file_path and os.path.exists(file_path):
                            os.remove(file_path)
                        
                        self._version_repo.delete(version['id'])
                        cleaned_count += 1
            
            return cleaned_count
            
        except Exception as e:
            logger.error(f"Failed to cleanup old versions: {e}")
            return 0
    
    # ==================== 私有辅助方法 ====================
    
    def _init_file_type_mapping(self) -> Dict[str, FileType]:
        """初始化文件类型映射"""
        return {
            # 文档
            '.txt': FileType.DOCUMENT,
            '.doc': FileType.DOCUMENT,
            '.docx': FileType.DOCUMENT,
            '.pdf': FileType.DOCUMENT,
            '.rtf': FileType.DOCUMENT,
            '.odt': FileType.DOCUMENT,
            # 图像
            '.jpg': FileType.IMAGE,
            '.jpeg': FileType.IMAGE,
            '.png': FileType.IMAGE,
            '.gif': FileType.IMAGE,
            '.bmp': FileType.IMAGE,
            '.svg': FileType.IMAGE,
            # 视频
            '.mp4': FileType.VIDEO,
            '.avi': FileType.VIDEO,
            '.mov': FileType.VIDEO,
            '.wmv': FileType.VIDEO,
            '.flv': FileType.VIDEO,
            # 音频
            '.mp3': FileType.AUDIO,
            '.wav': FileType.AUDIO,
            '.flac': FileType.AUDIO,
            '.aac': FileType.AUDIO,
            # 压缩包
            '.zip': FileType.ARCHIVE,
            '.rar': FileType.ARCHIVE,
            '.7z': FileType.ARCHIVE,
            '.tar': FileType.ARCHIVE,
            '.gz': FileType.ARCHIVE,
            # 代码
            '.py': FileType.CODE,
            '.js': FileType.CODE,
            '.html': FileType.CODE,
            '.css': FileType.CODE,
            '.java': FileType.CODE,
            '.cpp': FileType.CODE,
            '.c': FileType.CODE,
            # 数据
            '.csv': FileType.DATA,
            '.json': FileType.DATA,
            '.xml': FileType.DATA,
            '.yaml': FileType.DATA,
            '.yml': FileType.DATA,
            # 模型
            '.pkl': FileType.MODEL,
            '.h5': FileType.MODEL,
            '.pb': FileType.MODEL,
            '.onnx': FileType.MODEL,
            '.pth': FileType.MODEL,
            '.pt': FileType.MODEL,
            '.bin': FileType.MODEL,
            '.safetensors': FileType.MODEL,
        }
    
    def _init_default_policies(self):
        """初始化默认安全策略"""
        default_policies = [
            SecurityPolicy(
                id="public_default",
                name="公开文件默认策略",
                description="适用于公开文件的默认安全策略",
                security_level=SecurityLevel.PUBLIC,
                allowed_file_types=['.txt', '.pdf', '.jpg', '.png', '.csv', '.json'],
                max_file_size=10 * 1024 * 1024,
                encryption_required=False,
                virus_scan_required=True,
                access_control_enabled=False,
                audit_enabled=True,
                retention_days=365,
                created_at=datetime.now(),
                updated_at=datetime.now(),
                is_default=True
            ),
            SecurityPolicy(
                id="internal_default",
                name="内部文件默认策略",
                description="适用于内部文件的默认安全策略",
                security_level=SecurityLevel.INTERNAL,
                allowed_file_types=['.txt', '.pdf', '.doc', '.docx', '.csv', '.json', '.py', '.zip'],
                max_file_size=100 * 1024 * 1024,
                encryption_required=False,
                virus_scan_required=True,
                access_control_enabled=True,
                audit_enabled=True,
                retention_days=1095,
                created_at=datetime.now(),
                updated_at=datetime.now(),
                is_default=True
            ),
            SecurityPolicy(
                id="confidential_default",
                name="机密文件默认策略",
                description="适用于机密文件的默认安全策略",
                security_level=SecurityLevel.CONFIDENTIAL,
                allowed_file_types=['.txt', '.pdf', '.doc', '.docx', '.csv', '.json'],
                max_file_size=50 * 1024 * 1024,
                encryption_required=True,
                virus_scan_required=True,
                access_control_enabled=True,
                audit_enabled=True,
                retention_days=2555,
                created_at=datetime.now(),
                updated_at=datetime.now(),
                is_default=True
            ),
            SecurityPolicy(
                id="restricted_default",
                name="限制文件默认策略",
                description="适用于限制文件的默认安全策略",
                security_level=SecurityLevel.RESTRICTED,
                allowed_file_types=['.txt', '.pdf'],
                max_file_size=10 * 1024 * 1024,
                encryption_required=True,
                virus_scan_required=True,
                access_control_enabled=True,
                audit_enabled=True,
                retention_days=3650,
                created_at=datetime.now(),
                updated_at=datetime.now(),
                is_default=True
            )
        ]
        
        for policy in default_policies:
            self.create_security_policy(policy)
    
    def _detect_file_type(self, file_path: str) -> FileType:
        """检测文件类型"""
        _, ext = os.path.splitext(file_path.lower())
        return self.file_type_mapping.get(ext, FileType.UNKNOWN)
    
    def _get_applicable_policy(
        self,
        artifact_type: ArtifactType,
        security_level: SecurityLevel,
        tenant_id: str = None
    ) -> Optional[SecurityPolicy]:
        """获取适用的安全策略"""
        try:
            if not self._policy_repo:
                return None
            
            level_value = security_level.value if isinstance(security_level, SecurityLevel) else security_level
            
            # 首先查找特定策略
            policy_id = f"{artifact_type.value}_{level_value}"
            policy_data = self._policy_repo.get_by_id(policy_id, tenant_id)
            if policy_data:
                return self._dict_to_policy(policy_data)
            
            # 查找默认策略
            policy_data = self._policy_repo.get_default_for_level(level_value, tenant_id)
            if policy_data:
                return self._dict_to_policy(policy_data)
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get applicable policy: {e}")
            return None
    
    def _validate_file_type(self, file_path: str, policy: SecurityPolicy) -> bool:
        """验证文件类型"""
        _, ext = os.path.splitext(file_path.lower())
        return ext in policy.allowed_file_types
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """计算文件哈希"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def _perform_security_scan(self, file_path: str) -> SecurityScanResult:
        """执行安全扫描"""
        scan_result = SecurityScanResult(
            file_id="",
            scan_type="virus_scan",
            status="passed",
            threats_found=[],
            scan_time=datetime.now(),
            scanner_version="mock_1.0",
            details={}
        )
        
        # 简单的文件名检查
        filename = os.path.basename(file_path).lower()
        suspicious_patterns = ['virus', 'malware', 'trojan', 'backdoor']
        
        for pattern in suspicious_patterns:
            if pattern in filename:
                scan_result.status = "failed"
                scan_result.threats_found.append(f"Suspicious filename pattern: {pattern}")
        
        return scan_result
    
    def _encrypt_file(self, file_path: str) -> Optional[str]:
        """加密文件"""
        try:
            encrypted_path = file_path + ".encrypted"
            key = b"vectorsphere_key"
            
            with open(file_path, 'rb') as infile, open(encrypted_path, 'wb') as outfile:
                while True:
                    chunk = infile.read(1024)
                    if not chunk:
                        break
                    encrypted_chunk = bytes(a ^ b for a, b in zip(chunk, key * (len(chunk) // len(key) + 1)))
                    outfile.write(encrypted_chunk)
            
            return encrypted_path
            
        except Exception as e:
            logger.error(f"File encryption failed: {e}")
            return None
    
    def _set_file_permissions(self, file_path: str, security_level: SecurityLevel):
        """设置文件权限"""
        try:
            level = security_level if isinstance(security_level, SecurityLevel) else SecurityLevel(security_level)
            
            if level == SecurityLevel.PUBLIC:
                os.chmod(file_path, 0o644)
            elif level == SecurityLevel.INTERNAL:
                os.chmod(file_path, 0o640)
            elif level == SecurityLevel.CONFIDENTIAL:
                os.chmod(file_path, 0o600)
            elif level == SecurityLevel.RESTRICTED:
                os.chmod(file_path, 0o600)
        except Exception as e:
            logger.error(f"Failed to set file permissions: {e}")
    
    def _log_operation(
        self,
        operation: str,
        resource_type: str,
        resource_id: str,
        user_id: str,
        result: str,
        tenant_id: str = None
    ):
        """记录操作日志"""
        try:
            if self._access_log_repo:
                log_data = {
                    'tenant_id': tenant_id,
                    'artifact_id': resource_id if resource_type == 'artifact' else None,
                    'file_id': resource_id if resource_type == 'file' else None,
                    'user_id': user_id,
                    'operation': operation,
                    'result': result,
                    'details': {
                        'resource_type': resource_type,
                        'resource_id': resource_id
                    }
                }
                self._access_log_repo.create(log_data)
        except Exception as e:
            logger.error(f"Failed to log operation: {e}")
    
    def _dict_to_policy(self, data: Dict) -> SecurityPolicy:
        """字典转策略对象"""
        security_level = data.get('security_level')
        if isinstance(security_level, str):
            security_level = SecurityLevel(security_level)
        
        return SecurityPolicy(
            id=data['id'],
            name=data['name'],
            description=data.get('description', ''),
            security_level=security_level,
            allowed_file_types=data.get('allowed_file_types', []),
            max_file_size=data.get('max_file_size', 10485760),
            encryption_required=data.get('encryption_required', False),
            virus_scan_required=data.get('virus_scan_required', True),
            access_control_enabled=data.get('access_control_enabled', True),
            audit_enabled=data.get('audit_enabled', True),
            retention_days=data.get('retention_days', 365),
            created_at=data.get('created_at', datetime.now()),
            updated_at=data.get('updated_at', datetime.now()),
            is_default=data.get('is_default', False),
            is_active=data.get('is_active', True),
            tenant_id=data.get('tenant_id'),
            created_by=data.get('created_by')
        )
    
    def _dict_to_artifact(self, data: Dict) -> Artifact:
        """字典转工件对象"""
        artifact_type = data.get('artifact_type')
        if isinstance(artifact_type, str):
            artifact_type = ArtifactType(artifact_type)
        
        security_level = data.get('security_level')
        if isinstance(security_level, str):
            security_level = SecurityLevel(security_level)
        
        status = data.get('status', 'active')
        if isinstance(status, str):
            status = ArtifactStatus(status)
        
        return Artifact(
            id=data['id'],
            name=data['name'],
            description=data.get('description', ''),
            artifact_type=artifact_type,
            security_level=security_level,
            status=status,
            owner_id=data['owner_id'],
            current_version=data.get('current_version'),
            versions=[],
            tags=data.get('tags', []),
            metadata=data.get('metadata', {}),
            created_at=data.get('created_at', datetime.now()),
            updated_at=data.get('updated_at', datetime.now()),
            tenant_id=data.get('tenant_id'),
            version_count=data.get('version_count', 0),
            total_size=data.get('total_size', 0),
            policy_id=data.get('policy_id')
        )
    
    def _dict_to_version(self, data: Dict) -> ArtifactVersion:
        """字典转版本对象"""
        return ArtifactVersion(
            id=data['id'],
            artifact_id=data['artifact_id'],
            version=data['version'],
            status=data.get('status', 'active'),
            file_path=data.get('file_path'),
            file_size=data.get('file_size', 0),
            file_hash=data.get('file_hash'),
            mime_type=data.get('mime_type'),
            changelog=data.get('changelog', ''),
            tags=data.get('tags', []),
            metadata=data.get('metadata', {}),
            created_by=data['created_by'],
            created_at=data.get('created_at', datetime.now())
        )
    
    def _file_metadata_to_dict(self, metadata: FileMetadata) -> Dict:
        """文件元数据转字典"""
        return {
            'id': metadata.id,
            'tenant_id': metadata.tenant_id,
            'original_name': metadata.original_name,
            'stored_name': metadata.stored_name,
            'file_path': metadata.file_path,
            'file_type': metadata.file_type.value if isinstance(metadata.file_type, FileType) else metadata.file_type,
            'mime_type': metadata.mime_type,
            'size': metadata.size,
            'hash_sha256': metadata.hash_sha256,
            'security_level': metadata.security_level.value if isinstance(metadata.security_level, SecurityLevel) else metadata.security_level,
            'artifact_type': metadata.artifact_type.value if isinstance(metadata.artifact_type, ArtifactType) else metadata.artifact_type,
            'artifact_id': metadata.artifact_id,
            'version_id': metadata.version_id,
            'owner_id': metadata.owner_id,
            'is_encrypted': metadata.is_encrypted,
            'encryption_key_id': metadata.encryption_key_id,
            'tags': metadata.tags,
            'metadata': metadata.metadata
        }


# ==================== 工件管理服务 ====================

class ArtifactManagementService:
    """工件管理服务
    
    提供工件的高级管理功能，委托给 ArtifactSecurityService 处理核心逻辑。
    """
    
    def __init__(self, config: Dict[str, Any] = None, security_service: ArtifactSecurityService = None):
        self.config = config or {}
        self._security_service = security_service or ArtifactSecurityService(config)
    
    def create_artifact(
        self,
        name: str,
        description: str,
        artifact_type: ArtifactType,
        security_level: SecurityLevel,
        owner_id: str,
        tags: List[str] = None,
        metadata: Dict[str, Any] = None,
        tenant_id: str = None
    ) -> Tuple[bool, str, Optional[Artifact]]:
        """创建工件"""
        return self._security_service.create_artifact(
            name=name,
            description=description,
            artifact_type=artifact_type,
            security_level=security_level,
            owner_id=owner_id,
            tags=tags,
            metadata=metadata,
            tenant_id=tenant_id
        )
    
    def get_artifact(self, artifact_id: str, user_id: str, tenant_id: str = None) -> Optional[Artifact]:
        """获取工件"""
        return self._security_service.get_artifact(artifact_id, user_id, tenant_id)
    
    def list_artifacts(self, user_id: str, filters: Dict = None, tenant_id: str = None) -> List[Artifact]:
        """列出工件"""
        return self._security_service.list_artifacts(user_id, filters, tenant_id)
    
    def upload_artifact_version(
        self,
        artifact_id: str,
        file_path: str,
        version: str,
        changelog: str,
        user_id: str,
        tags: List[str] = None
    ) -> Tuple[bool, str, Optional[ArtifactVersion]]:
        """上传工件版本"""
        return self._security_service.upload_artifact_version(
            artifact_id=artifact_id,
            file_path=file_path,
            version=version,
            changelog=changelog,
            user_id=user_id,
            tags=tags
        )
    
    def get_artifact_versions(self, artifact_id: str, user_id: str) -> List[ArtifactVersion]:
        """获取工件版本列表"""
        return self._security_service.get_artifact_versions(artifact_id, user_id)
    
    def download_artifact_version(
        self,
        artifact_id: str,
        version: str,
        user_id: str
    ) -> Tuple[bool, str, Optional[str]]:
        """下载工件版本"""
        return self._security_service.download_artifact_version(artifact_id, version, user_id)
    
    def delete_artifact_version(
        self,
        artifact_id: str,
        version: str,
        user_id: str
    ) -> Tuple[bool, str]:
        """删除工件版本"""
        return self._security_service.delete_artifact_version(artifact_id, version, user_id)
    
    def update_artifact_status(
        self,
        artifact_id: str,
        status: ArtifactStatus,
        user_id: str
    ) -> Tuple[bool, str]:
        """更新工件状态"""
        return self._security_service.update_artifact_status(artifact_id, status, user_id)
    
    def add_dependency(
        self,
        source_id: str,
        target_id: str,
        dependency_type: str,
        version_constraint: str,
        user_id: str
    ) -> Tuple[bool, str]:
        """添加依赖"""
        return self._security_service.add_dependency(
            source_id, target_id, dependency_type, version_constraint, user_id
        )
    
    def remove_dependency(self, source_id: str, target_id: str, user_id: str) -> Tuple[bool, str]:
        """移除依赖"""
        return self._security_service.remove_dependency(source_id, target_id, user_id)
    
    def get_artifact_dependencies(self, artifact_id: str) -> List[ArtifactDependency]:
        """获取依赖列表"""
        return self._security_service.get_artifact_dependencies(artifact_id)
    
    def get_artifact_dependents(self, artifact_id: str) -> List[ArtifactDependency]:
        """获取被依赖列表"""
        return self._security_service.get_artifact_dependents(artifact_id)
    
    def cleanup_old_versions(self, retention_days: int = 90) -> int:
        """清理旧版本"""
        return self._security_service.cleanup_old_versions(retention_days)


# ==================== 单例获取函数 ====================

_artifact_security_service = None


def get_artifact_security_service(config: Dict[str, Any] = None, use_memory: bool = False) -> ArtifactSecurityService:
    """获取工件安全服务实例"""
    global _artifact_security_service
    if _artifact_security_service is None:
        _artifact_security_service = ArtifactSecurityService(config, use_memory)
    return _artifact_security_service


def reset_artifact_security_service():
    """重置服务实例（用于测试）"""
    global _artifact_security_service
    _artifact_security_service = None

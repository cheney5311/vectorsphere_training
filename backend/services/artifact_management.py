#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""工件管理与版本控制服务

提供工件的版本控制、生命周期管理、依赖关系管理等功能。
使用仓库层进行数据持久化。
"""

import logging
import os
import shutil
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple

logger = logging.getLogger(__name__)


# ==================== 枚举类型 ====================

class ArtifactStatus(Enum):
    """工件状态"""
    DRAFT = "draft"
    PENDING_REVIEW = "pending_review"
    APPROVED = "approved"
    PUBLISHED = "published"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"
    ACTIVE = "active"
    DELETED = "deleted"
    PENDING = "pending"
    LOCKED = "locked"


class VersionStatus(Enum):
    """版本状态"""
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    DELETED = "deleted"


class DependencyType(Enum):
    """依赖类型"""
    REQUIRED = "required"
    OPTIONAL = "optional"
    DEVELOPMENT = "development"


# ==================== 数据类 ====================

@dataclass
class ArtifactVersion:
    """工件版本"""
    id: str
    artifact_id: str
    version: str
    file_metadata_id: str
    file_path: str
    file_size: int
    file_hash: str
    mime_type: str
    status: VersionStatus
    changelog: str
    created_by: str
    created_at: datetime
    tags: List[str]
    metadata: Dict[str, Any]


@dataclass
class Artifact:
    """工件"""
    id: str
    name: str
    description: str
    artifact_type: str
    security_level: str
    status: ArtifactStatus
    owner_id: str
    current_version: Optional[str]
    versions: List[ArtifactVersion]
    dependencies: List[str]
    tags: List[str]
    metadata: Dict[str, Any]
    created_at: datetime
    updated_at: datetime
    tenant_id: str = None
    version_count: int = 0
    total_size: int = 0
    policy_id: str = None


@dataclass
class ArtifactDependency:
    """工件依赖关系"""
    id: str
    source_artifact_id: str
    target_artifact_id: str
    dependency_type: str
    version_constraint: str
    created_by: str
    created_at: datetime


# ==================== 服务类 ====================

class ArtifactManagementService:
    """工件管理服务
    
    提供工件的版本控制、生命周期管理、依赖关系管理等功能。
    委托 ArtifactSecurityService 处理安全相关逻辑。
    """
    
    def __init__(self, config: Dict[str, Any] = None, security_service=None, use_memory_storage: bool = False):
        self.config = config or {}
        self._use_memory_storage = use_memory_storage
        
        # 初始化安全服务
        self._security_service = security_service
        if self._security_service is None:
            self._init_security_service()
        
        # 初始化仓库
        self._init_repositories()
        
        # 存储路径
        self.storage_base_path = self.config.get('artifact_storage_path', '/var/lib/vectorsphere/artifacts')
        os.makedirs(self.storage_base_path, exist_ok=True)
    
    def _init_security_service(self):
        """初始化安全服务"""
        try:
            from backend.services.artifact_security import get_artifact_security_service
            self._security_service = get_artifact_security_service(self.config, self._use_memory_storage)
        except ImportError as e:
            logger.warning(f"Failed to import ArtifactSecurityService: {e}")
            self._security_service = None
    
    def _init_repositories(self):
        """初始化仓库"""
        try:
            from backend.repositories.artifact_repository import (
                get_artifact_repository,
                get_artifact_version_repository,
                get_artifact_dependency_repository,
                get_artifact_access_log_repository
            )
            self._artifact_repo = get_artifact_repository(use_memory=self._use_memory_storage)
            self._version_repo = get_artifact_version_repository(use_memory=self._use_memory_storage)
            self._dependency_repo = get_artifact_dependency_repository(use_memory=self._use_memory_storage)
            self._access_log_repo = get_artifact_access_log_repository(use_memory=self._use_memory_storage)
        except ImportError as e:
            logger.warning(f"Failed to import repositories: {e}")
            self._artifact_repo = None
            self._version_repo = None
            self._dependency_repo = None
            self._access_log_repo = None
    
    @property
    def security_service(self):
        """获取安全服务"""
        return self._security_service
    
    # ==================== 工件创建和管理 ====================
    
    def create_artifact(
        self,
        name: str,
        description: str,
        artifact_type: str,
        security_level: str,
        owner_id: str,
        tags: List[str] = None,
        metadata: Dict[str, Any] = None,
        tenant_id: str = None,
        policy_id: str = None
    ) -> Tuple[bool, str, Optional[Artifact]]:
        """创建工件
        
        Args:
            name: 工件名称
            description: 工件描述
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
            
            # 检查名称是否已存在
            existing = self._artifact_repo.get_all(
                tenant_id=tenant_id,
                owner_id=owner_id,
                limit=1000
            )
            for artifact in existing:
                if artifact.get('name') == name:
                    return False, "Artifact name already exists", None
            
            # 创建工件数据
            artifact_data = {
                'tenant_id': tenant_id,
                'name': name,
                'description': description,
                'artifact_type': artifact_type,
                'security_level': security_level,
                'status': ArtifactStatus.DRAFT.value,
                'owner_id': owner_id,
                'tags': tags or [],
                'metadata': metadata or {},
                'policy_id': policy_id
            }
            
            # 保存到仓库
            result = self._artifact_repo.create(artifact_data)
            if not result:
                return False, "Failed to create artifact", None
            
            # 转换为对象
            artifact = self._dict_to_artifact(result)
            
            # 记录操作日志
            self._log_operation("create", artifact.id, owner_id, tenant_id, {
                'name': name,
                'artifact_type': artifact_type
            })
            
            return True, "Artifact created successfully", artifact
            
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
            if not self._can_access_artifact(artifact_data, user_id):
                return None
            
            artifact = self._dict_to_artifact(artifact_data)
            
            # 加载版本列表
            if self._version_repo:
                versions_data = self._version_repo.get_by_artifact(artifact_id)
                artifact.versions = [self._dict_to_version(v) for v in versions_data]
            
            # 加载依赖列表
            if self._dependency_repo:
                deps_data = self._dependency_repo.get_dependencies(artifact_id)
                artifact.dependencies = [d['target_artifact_id'] for d in deps_data]
            
            return artifact
            
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
            for data in artifacts_data:
                # 检查访问权限
                if not self._can_access_artifact(data, user_id):
                    continue
                
                # 应用额外过滤条件
                if filters.get('tags'):
                    artifact_tags = data.get('tags', [])
                    if not any(tag in artifact_tags for tag in filters['tags']):
                        continue
                
                result.append(self._dict_to_artifact(data))
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to list artifacts: {e}")
            return []
    
    def update_artifact(
        self,
        artifact_id: str,
        updates: Dict[str, Any],
        user_id: str,
        tenant_id: str = None
    ) -> Tuple[bool, str]:
        """更新工件
        
        Args:
            artifact_id: 工件ID
            updates: 更新内容
            user_id: 用户ID
            tenant_id: 租户ID
            
        Returns:
            (是否成功, 错误信息)
        """
        try:
            if not self._artifact_repo:
                return False, "Repository not available"
            
            # 获取现有工件
            artifact_data = self._artifact_repo.get_by_id(artifact_id, tenant_id)
            if not artifact_data:
                return False, "Artifact not found"
            
            # 检查权限
            if artifact_data.get('owner_id') != user_id:
                return False, "Permission denied"
            
            # 过滤可更新字段
            allowed_fields = {'name', 'description', 'tags', 'metadata', 'policy_id'}
            filtered_updates = {k: v for k, v in updates.items() if k in allowed_fields}
            
            if not filtered_updates:
                return False, "No valid fields to update"
            
            # 更新
            success = self._artifact_repo.update(artifact_id, filtered_updates, tenant_id)
            
            if success:
                self._log_operation("update", artifact_id, user_id, tenant_id, filtered_updates)
                return True, "Artifact updated successfully"
            
            return False, "Update failed"
            
        except Exception as e:
            logger.error(f"Failed to update artifact: {e}")
            return False, f"Update failed: {str(e)}"
    
    def delete_artifact(
        self,
        artifact_id: str,
        user_id: str,
        tenant_id: str = None
    ) -> Tuple[bool, str]:
        """删除工件
        
        Args:
            artifact_id: 工件ID
            user_id: 用户ID
            tenant_id: 租户ID
            
        Returns:
            (是否成功, 错误信息)
        """
        try:
            if not self._artifact_repo:
                return False, "Repository not available"
            
            # 获取工件
            artifact_data = self._artifact_repo.get_by_id(artifact_id, tenant_id)
            if not artifact_data:
                return False, "Artifact not found"
            
            # 检查权限
            if artifact_data.get('owner_id') != user_id:
                return False, "Permission denied"
            
            # 检查是否有依赖此工件的其他工件
            if self._dependency_repo:
                dependents = self._dependency_repo.get_dependents(artifact_id)
                if dependents:
                    return False, f"Cannot delete: {len(dependents)} artifacts depend on this"
            
            # 删除所有版本
            if self._version_repo:
                versions = self._version_repo.get_by_artifact(artifact_id)
                for version in versions:
                    # 删除版本文件
                    file_path = version.get('file_path')
                    if file_path and os.path.exists(file_path):
                        os.remove(file_path)
                    self._version_repo.delete(version['id'])
            
            # 删除依赖关系
            if self._dependency_repo:
                deps = self._dependency_repo.get_dependencies(artifact_id)
                for dep in deps:
                    self._dependency_repo.delete(artifact_id, dep['target_artifact_id'])
            
            # 删除工件
            success = self._artifact_repo.delete(artifact_id, tenant_id)
            
            if success:
                self._log_operation("delete", artifact_id, user_id, tenant_id)
                return True, "Artifact deleted successfully"
            
            return False, "Deletion failed"
            
        except Exception as e:
            logger.error(f"Failed to delete artifact: {e}")
            return False, f"Deletion failed: {str(e)}"
    
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
            
            # 获取工件
            artifact_data = self._artifact_repo.get_by_id(artifact_id, tenant_id)
            if not artifact_data:
                return False, "Artifact not found"
            
            # 检查权限
            if artifact_data.get('owner_id') != user_id:
                return False, "Permission denied"
            
            old_status = artifact_data.get('status')
            status_value = status.value if isinstance(status, ArtifactStatus) else status
            
            # 验证状态转换
            if not self._is_valid_status_transition(old_status, status_value):
                return False, f"Invalid status transition from {old_status} to {status_value}"
            
            # 更新状态
            success = self._artifact_repo.update_status(artifact_id, status_value, tenant_id)
            
            if success:
                self._log_operation("update_status", artifact_id, user_id, tenant_id, {
                    'old_status': old_status,
                    'new_status': status_value
                })
                return True, "Status updated successfully"
            
            return False, "Status update failed"
            
        except Exception as e:
            logger.error(f"Failed to update artifact status: {e}")
            return False, f"Status update failed: {str(e)}"
    
    # ==================== 版本管理 ====================
    
    def upload_artifact_version(
        self,
        artifact_id: str,
        file_path: str,
        version: str,
        changelog: str,
        user_id: str,
        tags: List[str] = None,
        metadata: Dict[str, Any] = None,
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
            metadata: 元数据
            tenant_id: 租户ID
            
        Returns:
            (是否成功, 错误信息, 版本对象)
        """
        try:
            if not self._artifact_repo or not self._version_repo:
                return False, "Repository not available", None
            
            # 获取工件
            artifact_data = self._artifact_repo.get_by_id(artifact_id, tenant_id)
            if not artifact_data:
                return False, "Artifact not found", None
            
            # 检查权限
            if artifact_data.get('owner_id') != user_id:
                return False, "Permission denied", None
            
            # 检查版本是否已存在
            existing_version = self._version_repo.get_by_artifact_and_version(artifact_id, version)
            if existing_version:
                return False, "Version already exists", None
            
            # 验证文件
            if not os.path.exists(file_path):
                return False, "File not found", None
            
            # 使用安全服务验证和存储文件
            if self._security_service:
                from backend.services.artifact_security import ArtifactType, SecurityLevel
                
                try:
                    artifact_type = ArtifactType(artifact_data['artifact_type'])
                    security_level = SecurityLevel(artifact_data['security_level'])
                except ValueError:
                    artifact_type = artifact_data['artifact_type']
                    security_level = artifact_data['security_level']
                
                success, msg, file_metadata = self._security_service.validate_file_upload(
                    file_path, user_id, artifact_type, security_level, tenant_id
                )
                if not success:
                    return False, msg, None
                
                success, storage_path = self._security_service.store_secure_file(
                    file_metadata, file_path, tenant_id
                )
                if not success:
                    return False, storage_path, None
                
                file_id = file_metadata.id
                stored_path = storage_path
                file_size = file_metadata.size
                file_hash = file_metadata.hash_sha256
                mime_type = file_metadata.mime_type
            else:
                # 简化处理：直接存储文件
                import hashlib
                import mimetypes
                
                file_size = os.path.getsize(file_path)
                with open(file_path, 'rb') as f:
                    file_hash = hashlib.sha256(f.read()).hexdigest()
                mime_type = mimetypes.guess_type(file_path)[0] or 'application/octet-stream'
                
                # 创建存储目录
                storage_dir = os.path.join(
                    self.storage_base_path,
                    artifact_id,
                    version
                )
                os.makedirs(storage_dir, exist_ok=True)
                
                stored_path = os.path.join(storage_dir, os.path.basename(file_path))
                shutil.copy2(file_path, stored_path)
                file_id = str(uuid.uuid4())
            
            # 创建版本记录
            version_data = {
                'artifact_id': artifact_id,
                'version': version,
                'status': VersionStatus.ACTIVE.value,
                'file_path': stored_path,
                'file_size': file_size,
                'file_hash': file_hash,
                'mime_type': mime_type,
                'changelog': changelog,
                'tags': tags or [],
                'metadata': metadata or {},
                'created_by': user_id
            }
            
            version_result = self._version_repo.create(version_data)
            if not version_result:
                return False, "Failed to create version record", None
            
            # 更新工件
            self._artifact_repo.update(artifact_id, {'current_version': version}, tenant_id)
            self._artifact_repo.increment_version_count(artifact_id, file_size)
            
            # 转换为对象
            artifact_version = self._dict_to_version(version_result)
            artifact_version.file_metadata_id = file_id
            
            # 记录日志
            self._log_operation("upload_version", artifact_id, user_id, tenant_id, {
                'version': version,
                'file_size': file_size
            })
            
            return True, "Version uploaded successfully", artifact_version
            
        except Exception as e:
            logger.error(f"Failed to upload artifact version: {e}")
            return False, f"Upload failed: {str(e)}", None
    
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
            if not self._artifact_repo or not self._version_repo:
                return False, "Repository not available", None
            
            # 获取工件
            artifact_data = self._artifact_repo.get_by_id(artifact_id, tenant_id)
            if not artifact_data:
                return False, "Artifact not found", None
            
            # 检查访问权限
            if not self._can_access_artifact(artifact_data, user_id):
                return False, "Permission denied", None
            
            # 获取版本
            version_data = self._version_repo.get_by_artifact_and_version(artifact_id, version)
            if not version_data:
                return False, "Version not found", None
            
            # 检查版本状态
            if version_data.get('status') == VersionStatus.DELETED.value:
                return False, "Version has been deleted", None
            
            # 检查文件
            file_path = version_data.get('file_path')
            if not file_path or not os.path.exists(file_path):
                return False, "File not found or damaged", None
            
            # 记录下载日志
            self._log_operation("download", artifact_id, user_id, tenant_id, {
                'version': version
            })
            
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
            if not self._artifact_repo or not self._version_repo:
                return False, "Repository not available"
            
            # 获取工件
            artifact_data = self._artifact_repo.get_by_id(artifact_id, tenant_id)
            if not artifact_data:
                return False, "Artifact not found"
            
            # 检查权限
            if artifact_data.get('owner_id') != user_id:
                return False, "Permission denied"
            
            # 获取版本
            version_data = self._version_repo.get_by_artifact_and_version(artifact_id, version)
            if not version_data:
                return False, "Version not found"
            
            # 检查是否是当前版本
            if artifact_data.get('current_version') == version:
                # 找到下一个可用版本
                all_versions = self._version_repo.get_by_artifact(artifact_id, status=VersionStatus.ACTIVE.value)
                available = [v for v in all_versions if v['version'] != version]
                
                if available:
                    # 选择最新的版本
                    latest = max(available, key=lambda v: v.get('created_at', datetime.min))
                    self._artifact_repo.update(artifact_id, {'current_version': latest['version']}, tenant_id)
                else:
                    self._artifact_repo.update(artifact_id, {'current_version': None}, tenant_id)
            
            # 删除文件
            file_path = version_data.get('file_path')
            if file_path and os.path.exists(file_path):
                os.remove(file_path)
            
            # 标记版本为已删除
            self._version_repo.update_status(version_data['id'], VersionStatus.DELETED.value)
            
            # 记录日志
            self._log_operation("delete_version", artifact_id, user_id, tenant_id, {
                'version': version
            })
            
            return True, "Version deleted successfully"
            
        except Exception as e:
            logger.error(f"Failed to delete artifact version: {e}")
            return False, f"Deletion failed: {str(e)}"
    
    def get_artifact_versions(
        self,
        artifact_id: str,
        user_id: str,
        tenant_id: str = None,
        include_deleted: bool = False
    ) -> List[ArtifactVersion]:
        """获取工件版本列表
        
        Args:
            artifact_id: 工件ID
            user_id: 用户ID
            tenant_id: 租户ID
            include_deleted: 是否包含已删除版本
            
        Returns:
            版本列表
        """
        try:
            if not self._artifact_repo or not self._version_repo:
                return []
            
            # 检查工件访问权限
            artifact_data = self._artifact_repo.get_by_id(artifact_id, tenant_id)
            if not artifact_data or not self._can_access_artifact(artifact_data, user_id):
                return []
            
            # 获取版本
            status = None if include_deleted else VersionStatus.ACTIVE.value
            versions_data = self._version_repo.get_by_artifact(artifact_id, status)
            
            return [self._dict_to_version(v) for v in versions_data]
            
        except Exception as e:
            logger.error(f"Failed to get artifact versions: {e}")
            return []
    
    def compare_versions(
        self,
        artifact_id: str,
        version1: str,
        version2: str,
        user_id: str,
        tenant_id: str = None
    ) -> Dict[str, Any]:
        """比较两个版本
        
        Args:
            artifact_id: 工件ID
            version1: 版本1
            version2: 版本2
            user_id: 用户ID
            tenant_id: 租户ID
            
        Returns:
            比较结果
        """
        try:
            if not self._version_repo:
                return {'error': 'Repository not available'}
            
            v1_data = self._version_repo.get_by_artifact_and_version(artifact_id, version1)
            v2_data = self._version_repo.get_by_artifact_and_version(artifact_id, version2)
            
            if not v1_data or not v2_data:
                return {'error': 'One or both versions not found'}
            
            return {
                'version1': {
                    'version': v1_data['version'],
                    'file_size': v1_data.get('file_size', 0),
                    'file_hash': v1_data.get('file_hash'),
                    'created_at': v1_data.get('created_at'),
                    'changelog': v1_data.get('changelog')
                },
                'version2': {
                    'version': v2_data['version'],
                    'file_size': v2_data.get('file_size', 0),
                    'file_hash': v2_data.get('file_hash'),
                    'created_at': v2_data.get('created_at'),
                    'changelog': v2_data.get('changelog')
                },
                'size_diff': v2_data.get('file_size', 0) - v1_data.get('file_size', 0),
                'hash_changed': v1_data.get('file_hash') != v2_data.get('file_hash'),
                'is_newer': self._is_newer_version(version2, version1)
            }
            
        except Exception as e:
            logger.error(f"Failed to compare versions: {e}")
            return {'error': str(e)}
    
    # ==================== 依赖管理 ====================
    
    def add_dependency(
        self,
        source_artifact_id: str,
        target_artifact_id: str,
        dependency_type: str,
        version_constraint: str,
        user_id: str,
        tenant_id: str = None
    ) -> Tuple[bool, str]:
        """添加依赖关系
        
        Args:
            source_artifact_id: 源工件ID
            target_artifact_id: 目标工件ID
            dependency_type: 依赖类型
            version_constraint: 版本约束
            user_id: 用户ID
            tenant_id: 租户ID
            
        Returns:
            (是否成功, 错误信息)
        """
        try:
            if not self._artifact_repo or not self._dependency_repo:
                return False, "Repository not available"
            
            # 检查工件是否存在
            source = self._artifact_repo.get_by_id(source_artifact_id, tenant_id)
            if not source:
                return False, "Source artifact not found"
            
            target = self._artifact_repo.get_by_id(target_artifact_id, tenant_id)
            if not target:
                return False, "Target artifact not found"
            
            # 检查权限
            if source.get('owner_id') != user_id:
                return False, "Permission denied"
            
            # 检查自引用
            if source_artifact_id == target_artifact_id:
                return False, "Cannot add self-reference dependency"
            
            # 检查循环依赖
            if self._would_create_cycle(source_artifact_id, target_artifact_id):
                return False, "Would create circular dependency"
            
            # 检查是否已存在
            existing_deps = self._dependency_repo.get_dependencies(source_artifact_id)
            for dep in existing_deps:
                if dep['target_artifact_id'] == target_artifact_id:
                    return False, "Dependency already exists"
            
            # 创建依赖
            dep_data = {
                'source_artifact_id': source_artifact_id,
                'target_artifact_id': target_artifact_id,
                'dependency_type': dependency_type,
                'version_constraint': version_constraint,
                'created_by': user_id
            }
            
            result = self._dependency_repo.create(dep_data)
            if result:
                self._log_operation("add_dependency", source_artifact_id, user_id, tenant_id, {
                    'target': target_artifact_id,
                    'type': dependency_type
                })
                return True, "Dependency added successfully"
            
            return False, "Failed to add dependency"
            
        except Exception as e:
            logger.error(f"Failed to add dependency: {e}")
            return False, f"Failed: {str(e)}"
    
    def remove_dependency(
        self,
        source_artifact_id: str,
        target_artifact_id: str,
        user_id: str,
        tenant_id: str = None
    ) -> Tuple[bool, str]:
        """移除依赖关系
        
        Args:
            source_artifact_id: 源工件ID
            target_artifact_id: 目标工件ID
            user_id: 用户ID
            tenant_id: 租户ID
            
        Returns:
            (是否成功, 错误信息)
        """
        try:
            if not self._artifact_repo or not self._dependency_repo:
                return False, "Repository not available"
            
            # 检查权限
            source = self._artifact_repo.get_by_id(source_artifact_id, tenant_id)
            if not source:
                return False, "Source artifact not found"
            
            if source.get('owner_id') != user_id:
                return False, "Permission denied"
            
            # 删除依赖
            success = self._dependency_repo.delete(source_artifact_id, target_artifact_id)
            
            if success:
                self._log_operation("remove_dependency", source_artifact_id, user_id, tenant_id, {
                    'target': target_artifact_id
                })
                return True, "Dependency removed successfully"
            
            return False, "Dependency not found"
            
        except Exception as e:
            logger.error(f"Failed to remove dependency: {e}")
            return False, f"Failed: {str(e)}"
    
    def get_artifact_dependencies(
        self,
        artifact_id: str,
        user_id: str = None,
        tenant_id: str = None
    ) -> List[ArtifactDependency]:
        """获取工件的依赖列表
        
        Args:
            artifact_id: 工件ID
            user_id: 用户ID
            tenant_id: 租户ID
            
        Returns:
            依赖关系列表
        """
        try:
            if not self._dependency_repo:
                return []
            
            deps_data = self._dependency_repo.get_dependencies(artifact_id)
            return [self._dict_to_dependency(d) for d in deps_data]
            
        except Exception as e:
            logger.error(f"Failed to get dependencies: {e}")
            return []
    
    def get_artifact_dependents(
        self,
        artifact_id: str,
        user_id: str = None,
        tenant_id: str = None
    ) -> List[ArtifactDependency]:
        """获取依赖此工件的列表
        
        Args:
            artifact_id: 工件ID
            user_id: 用户ID
            tenant_id: 租户ID
            
        Returns:
            依赖关系列表
        """
        try:
            if not self._dependency_repo:
                return []
            
            deps_data = self._dependency_repo.get_dependents(artifact_id)
            return [self._dict_to_dependency(d) for d in deps_data]
            
        except Exception as e:
            logger.error(f"Failed to get dependents: {e}")
            return []
    
    def resolve_dependencies(
        self,
        artifact_id: str,
        tenant_id: str = None
    ) -> Dict[str, Any]:
        """解析工件的完整依赖树
        
        Args:
            artifact_id: 工件ID
            tenant_id: 租户ID
            
        Returns:
            依赖树
        """
        try:
            if not self._dependency_repo or not self._artifact_repo:
                return {'error': 'Repository not available'}
            
            resolved = {}
            visited = set()
            
            def resolve_recursive(art_id: str, depth: int = 0) -> Dict:
                if art_id in visited:
                    return {'circular_reference': True}
                
                if depth > 10:
                    return {'max_depth_exceeded': True}
                
                visited.add(art_id)
                
                artifact = self._artifact_repo.get_by_id(art_id, tenant_id)
                if not artifact:
                    return {'not_found': True}
                
                deps = self._dependency_repo.get_dependencies(art_id)
                children = {}
                
                for dep in deps:
                    target_id = dep['target_artifact_id']
                    children[target_id] = {
                        'dependency_type': dep['dependency_type'],
                        'version_constraint': dep['version_constraint'],
                        'dependencies': resolve_recursive(target_id, depth + 1)
                    }
                
                visited.remove(art_id)
                
                return {
                    'artifact_id': art_id,
                    'name': artifact.get('name'),
                    'current_version': artifact.get('current_version'),
                    'dependencies': children
                }
            
            return resolve_recursive(artifact_id)
            
        except Exception as e:
            logger.error(f"Failed to resolve dependencies: {e}")
            return {'error': str(e)}
    
    # ==================== 清理功能 ====================
    
    def cleanup_old_versions(
        self,
        retention_days: int = 90,
        keep_count: int = 3,
        tenant_id: str = None
    ) -> int:
        """清理旧版本
        
        Args:
            retention_days: 保留天数
            keep_count: 每个工件至少保留的版本数
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
                artifact_id = artifact['id']
                current_version = artifact.get('current_version')
                
                # 获取所有版本
                versions = self._version_repo.get_by_artifact(artifact_id)
                
                # 按创建时间排序
                versions.sort(key=lambda v: v.get('created_at', datetime.min), reverse=True)
                
                # 确定要保留的版本
                versions_to_keep = set()
                if current_version:
                    versions_to_keep.add(current_version)
                
                # 保留最近的N个版本
                for v in versions[:keep_count]:
                    versions_to_keep.add(v['version'])
                
                # 清理旧版本
                for version in versions:
                    ver = version['version']
                    created_at = version.get('created_at')
                    status = version.get('status')
                    
                    if (ver not in versions_to_keep and
                        status == VersionStatus.ACTIVE.value and
                        created_at and created_at < cutoff_date):
                        
                        # 删除文件
                        file_path = version.get('file_path')
                        if file_path and os.path.exists(file_path):
                            os.remove(file_path)
                        
                        # 标记为已删除
                        self._version_repo.update_status(version['id'], VersionStatus.DELETED.value)
                        
                        cleaned_count += 1
                        
                        self._log_operation("cleanup_version", artifact_id, "system", tenant_id, {
                            'version': ver,
                            'reason': 'retention_policy'
                        })
            
            return cleaned_count
            
        except Exception as e:
            logger.error(f"Failed to cleanup old versions: {e}")
            return 0
    
    # ==================== 私有辅助方法 ====================
    
    def _is_newer_version(self, version1: str, version2: str) -> bool:
        """比较版本号"""
        try:
            v1_parts = [int(x) for x in version1.split('.')]
            v2_parts = [int(x) for x in version2.split('.')]
            
            max_len = max(len(v1_parts), len(v2_parts))
            v1_parts.extend([0] * (max_len - len(v1_parts)))
            v2_parts.extend([0] * (max_len - len(v2_parts)))
            
            return v1_parts > v2_parts
        except ValueError:
            return version1 > version2
    
    def _would_create_cycle(self, source_id: str, target_id: str) -> bool:
        """检查是否会形成循环依赖"""
        if not self._dependency_repo:
            return False
        
        visited = set()
        
        def dfs(current_id: str) -> bool:
            if current_id == source_id:
                return True
            
            if current_id in visited:
                return False
            
            visited.add(current_id)
            
            deps = self._dependency_repo.get_dependencies(current_id)
            for dep in deps:
                if dfs(dep['target_artifact_id']):
                    return True
            
            return False
        
        return dfs(target_id)
    
    def _can_access_artifact(self, artifact_data: Dict, user_id: str) -> bool:
        """检查是否可以访问工件"""
        # 所有者可以访问
        if artifact_data.get('owner_id') == user_id:
            return True
        
        # 公开工件可以访问
        if artifact_data.get('security_level') == 'public':
            return True
        
        # TODO: 集成访问控制系统
        return False
    
    def _is_valid_status_transition(self, old_status: str, new_status: str) -> bool:
        """检查状态转换是否有效"""
        # 定义有效的状态转换
        valid_transitions = {
            'draft': ['pending_review', 'published', 'active', 'archived', 'deleted'],
            'pending_review': ['approved', 'draft', 'active', 'archived', 'deleted'],
            'approved': ['published', 'draft', 'active', 'archived', 'deleted'],
            'published': ['deprecated', 'archived', 'active', 'deleted'],
            'deprecated': ['archived', 'published', 'active', 'deleted'],
            'archived': ['draft', 'deleted'],
            'active': ['archived', 'deleted', 'deprecated', 'locked'],
            'pending': ['active', 'deleted'],
            'locked': ['active', 'archived', 'deleted'],
            'deleted': []
        }
        
        allowed = valid_transitions.get(old_status, [])
        return new_status in allowed
    
    def _log_operation(
        self,
        operation: str,
        artifact_id: str,
        user_id: str,
        tenant_id: str = None,
        details: Dict[str, Any] = None
    ):
        """记录操作日志"""
        try:
            if self._access_log_repo:
                log_data = {
                    'tenant_id': tenant_id,
                    'artifact_id': artifact_id,
                    'user_id': user_id,
                    'operation': operation,
                    'result': 'success',
                    'details': details or {}
                }
                self._access_log_repo.create(log_data)
            
            # 同时写入应用日志
            logger.info(f"Artifact operation: {operation} on {artifact_id} by {user_id}")
            
        except Exception as e:
            logger.warning(f"Failed to log operation: {e}")
    
    def _dict_to_artifact(self, data: Dict) -> Artifact:
        """字典转工件对象"""
        status = data.get('status', 'draft')
        if isinstance(status, str):
            try:
                status = ArtifactStatus(status)
            except ValueError:
                status = ArtifactStatus.DRAFT
        
        return Artifact(
            id=data['id'],
            name=data['name'],
            description=data.get('description', ''),
            artifact_type=data.get('artifact_type', ''),
            security_level=data.get('security_level', 'internal'),
            status=status,
            owner_id=data.get('owner_id', ''),
            current_version=data.get('current_version'),
            versions=[],
            dependencies=[],
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
        status = data.get('status', 'active')
        if isinstance(status, str):
            try:
                status = VersionStatus(status)
            except ValueError:
                status = VersionStatus.ACTIVE
        
        return ArtifactVersion(
            id=data['id'],
            artifact_id=data['artifact_id'],
            version=data['version'],
            file_metadata_id=data.get('file_metadata_id', ''),
            file_path=data.get('file_path', ''),
            file_size=data.get('file_size', 0),
            file_hash=data.get('file_hash', ''),
            mime_type=data.get('mime_type', ''),
            status=status,
            changelog=data.get('changelog', ''),
            created_by=data.get('created_by', ''),
            created_at=data.get('created_at', datetime.now()),
            tags=data.get('tags', []),
            metadata=data.get('metadata', {})
        )
    
    def _dict_to_dependency(self, data: Dict) -> ArtifactDependency:
        """字典转依赖对象"""
        return ArtifactDependency(
            id=data.get('id', ''),
            source_artifact_id=data['source_artifact_id'],
            target_artifact_id=data['target_artifact_id'],
            dependency_type=data.get('dependency_type', 'required'),
            version_constraint=data.get('version_constraint', '*'),
            created_by=data.get('created_by', ''),
            created_at=data.get('created_at', datetime.now())
        )


# ==================== 单例获取函数 ====================

_management_service = None


def get_artifact_management_service(
    config: Dict[str, Any] = None,
    use_memory: bool = False
) -> ArtifactManagementService:
    """获取工件管理服务实例"""
    global _management_service
    if _management_service is None:
        _management_service = ArtifactManagementService(config, use_memory_storage=use_memory)
    return _management_service


def reset_artifact_management_service():
    """重置服务实例（用于测试）"""
    global _management_service
    _management_service = None

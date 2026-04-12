# -*- coding: utf-8 -*-
"""工件安全仓库层

提供工件、安全策略、文件元数据等数据的持久化存储和访问。
"""

import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ==================== 基础仓库类 ====================

class BaseArtifactRepository:
    """工件仓库基类"""
    
    def __init__(self, db_service=None, use_memory: bool = False):
        self._db_service = db_service
        self._use_memory = use_memory or db_service is None
        
        if not self._use_memory:
            try:
                from backend.db.database_manager import get_database_manager
                self._db_manager = get_database_manager()
            except ImportError:
                logger.warning("Database manager not available, using memory storage")
                self._use_memory = True
                self._db_manager = None
        else:
            self._db_manager = None


# ==================== 安全策略仓库 ====================

class SecurityPolicyRepository(BaseArtifactRepository):
    """安全策略仓库"""
    
    # 内存存储
    _policies: Dict[str, Dict] = {}
    
    def create(self, policy_data: Dict[str, Any]) -> Optional[Dict]:
        """创建安全策略"""
        try:
            policy_id = policy_data.get('id') or f"policy_{uuid.uuid4().hex[:12]}"
            now = datetime.utcnow()
            
            policy = {
                'id': policy_id,
                'tenant_id': policy_data.get('tenant_id'),
                'name': policy_data['name'],
                'description': policy_data.get('description', ''),
                'security_level': policy_data['security_level'],
                'allowed_file_types': policy_data.get('allowed_file_types', []),
                'max_file_size': policy_data.get('max_file_size', 10485760),
                'encryption_required': policy_data.get('encryption_required', False),
                'virus_scan_required': policy_data.get('virus_scan_required', True),
                'access_control_enabled': policy_data.get('access_control_enabled', True),
                'audit_enabled': policy_data.get('audit_enabled', True),
                'retention_days': policy_data.get('retention_days', 365),
                'is_default': policy_data.get('is_default', False),
                'is_active': policy_data.get('is_active', True),
                'created_by': policy_data.get('created_by'),
                'created_at': now,
                'updated_at': now
            }
            
            if self._use_memory:
                self._policies[policy_id] = policy
                return policy
            
            # 数据库存储
            from backend.schemas.artifact_models import SecurityPolicyModel
            with self._db_manager.get_session() as session:
                model = SecurityPolicyModel(**policy)
                session.add(model)
                session.commit()
                return policy
                
        except Exception as e:
            logger.error(f"Failed to create security policy: {e}")
            return None
    
    def get_by_id(self, policy_id: str, tenant_id: str = None) -> Optional[Dict]:
        """获取安全策略"""
        try:
            if self._use_memory:
                policy = self._policies.get(policy_id)
                if policy and (tenant_id is None or policy.get('tenant_id') == tenant_id):
                    return policy
                return None
            
            from backend.schemas.artifact_models import SecurityPolicyModel
            with self._db_manager.get_session() as session:
                query = session.query(SecurityPolicyModel).filter(
                    SecurityPolicyModel.id == policy_id
                )
                if tenant_id:
                    query = query.filter(SecurityPolicyModel.tenant_id == tenant_id)
                model = query.first()
                if model:
                    return self._model_to_dict(model)
                return None
                
        except Exception as e:
            logger.error(f"Failed to get security policy: {e}")
            return None
    
    def get_all(self, tenant_id: str = None, security_level: str = None) -> List[Dict]:
        """获取所有安全策略"""
        try:
            if self._use_memory:
                policies = []
                for policy in self._policies.values():
                    if tenant_id and policy.get('tenant_id') != tenant_id:
                        continue
                    if security_level and policy.get('security_level') != security_level:
                        continue
                    policies.append(policy)
                return policies
            
            from backend.schemas.artifact_models import SecurityPolicyModel
            with self._db_manager.get_session() as session:
                query = session.query(SecurityPolicyModel)
                if tenant_id:
                    query = query.filter(SecurityPolicyModel.tenant_id == tenant_id)
                if security_level:
                    query = query.filter(SecurityPolicyModel.security_level == security_level)
                return [self._model_to_dict(m) for m in query.all()]
                
        except Exception as e:
            logger.error(f"Failed to get security policies: {e}")
            return []
    
    def update(self, policy_id: str, updates: Dict[str, Any], tenant_id: str = None) -> bool:
        """更新安全策略"""
        try:
            updates['updated_at'] = datetime.utcnow()
            
            if self._use_memory:
                if policy_id in self._policies:
                    policy = self._policies[policy_id]
                    if tenant_id and policy.get('tenant_id') != tenant_id:
                        return False
                    policy.update(updates)
                    return True
                return False
            
            from backend.schemas.artifact_models import SecurityPolicyModel
            with self._db_manager.get_session() as session:
                query = session.query(SecurityPolicyModel).filter(
                    SecurityPolicyModel.id == policy_id
                )
                if tenant_id:
                    query = query.filter(SecurityPolicyModel.tenant_id == tenant_id)
                result = query.update(updates)
                session.commit()
                return result > 0
                
        except Exception as e:
            logger.error(f"Failed to update security policy: {e}")
            return False
    
    def delete(self, policy_id: str, tenant_id: str = None) -> bool:
        """删除安全策略"""
        try:
            if self._use_memory:
                if policy_id in self._policies:
                    policy = self._policies[policy_id]
                    if tenant_id and policy.get('tenant_id') != tenant_id:
                        return False
                    del self._policies[policy_id]
                    return True
                return False
            
            from backend.schemas.artifact_models import SecurityPolicyModel
            with self._db_manager.get_session() as session:
                query = session.query(SecurityPolicyModel).filter(
                    SecurityPolicyModel.id == policy_id
                )
                if tenant_id:
                    query = query.filter(SecurityPolicyModel.tenant_id == tenant_id)
                result = query.delete()
                session.commit()
                return result > 0
                
        except Exception as e:
            logger.error(f"Failed to delete security policy: {e}")
            return False
    
    def get_default_for_level(self, security_level: str, tenant_id: str = None) -> Optional[Dict]:
        """获取指定安全级别的默认策略"""
        try:
            if self._use_memory:
                for policy in self._policies.values():
                    if policy.get('security_level') == security_level and policy.get('is_default'):
                        if tenant_id is None or policy.get('tenant_id') == tenant_id:
                            return policy
                return None
            
            from backend.schemas.artifact_models import SecurityPolicyModel
            with self._db_manager.get_session() as session:
                query = session.query(SecurityPolicyModel).filter(
                    SecurityPolicyModel.security_level == security_level,
                    SecurityPolicyModel.is_default == True
                )
                if tenant_id:
                    query = query.filter(SecurityPolicyModel.tenant_id == tenant_id)
                model = query.first()
                if model:
                    return self._model_to_dict(model)
                return None
                
        except Exception as e:
            logger.error(f"Failed to get default policy: {e}")
            return None
    
    def _model_to_dict(self, model) -> Dict:
        """模型转字典"""
        return {
            'id': model.id,
            'tenant_id': model.tenant_id,
            'name': model.name,
            'description': model.description,
            'security_level': model.security_level,
            'allowed_file_types': model.allowed_file_types or [],
            'max_file_size': model.max_file_size,
            'encryption_required': model.encryption_required,
            'virus_scan_required': model.virus_scan_required,
            'access_control_enabled': model.access_control_enabled,
            'audit_enabled': model.audit_enabled,
            'retention_days': model.retention_days,
            'is_default': model.is_default,
            'is_active': model.is_active,
            'created_by': model.created_by,
            'created_at': model.created_at,
            'updated_at': model.updated_at
        }


# ==================== 工件仓库 ====================

class ArtifactRepository(BaseArtifactRepository):
    """工件仓库"""
    
    # 内存存储
    _artifacts: Dict[str, Dict] = {}
    
    def create(self, artifact_data: Dict[str, Any]) -> Optional[Dict]:
        """创建工件"""
        try:
            artifact_id = artifact_data.get('id') or f"artifact_{uuid.uuid4().hex[:12]}"
            now = datetime.utcnow()
            
            artifact = {
                'id': artifact_id,
                'tenant_id': artifact_data.get('tenant_id'),
                'name': artifact_data['name'],
                'description': artifact_data.get('description', ''),
                'artifact_type': artifact_data['artifact_type'],
                'security_level': artifact_data['security_level'],
                'status': artifact_data.get('status', 'active'),
                'owner_id': artifact_data['owner_id'],
                'current_version': artifact_data.get('current_version'),
                'version_count': artifact_data.get('version_count', 0),
                'total_size': artifact_data.get('total_size', 0),
                'tags': artifact_data.get('tags', []),
                'metadata': artifact_data.get('metadata', {}),
                'policy_id': artifact_data.get('policy_id'),
                'created_at': now,
                'updated_at': now
            }
            
            if self._use_memory:
                self._artifacts[artifact_id] = artifact
                return artifact
            
            from backend.schemas.artifact_models import ArtifactModel
            with self._db_manager.get_session() as session:
                model = ArtifactModel(**artifact)
                session.add(model)
                session.commit()
                return artifact
                
        except Exception as e:
            logger.error(f"Failed to create artifact: {e}")
            return None
    
    def get_by_id(self, artifact_id: str, tenant_id: str = None) -> Optional[Dict]:
        """获取工件"""
        try:
            if self._use_memory:
                artifact = self._artifacts.get(artifact_id)
                if artifact and (tenant_id is None or artifact.get('tenant_id') == tenant_id):
                    return artifact
                return None
            
            from backend.schemas.artifact_models import ArtifactModel
            with self._db_manager.get_session() as session:
                query = session.query(ArtifactModel).filter(
                    ArtifactModel.id == artifact_id
                )
                if tenant_id:
                    query = query.filter(ArtifactModel.tenant_id == tenant_id)
                model = query.first()
                if model:
                    return self._model_to_dict(model)
                return None
                
        except Exception as e:
            logger.error(f"Failed to get artifact: {e}")
            return None
    
    def get_all(
        self,
        tenant_id: str = None,
        owner_id: str = None,
        artifact_type: str = None,
        security_level: str = None,
        status: str = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict]:
        """获取工件列表"""
        try:
            if self._use_memory:
                artifacts = []
                for artifact in self._artifacts.values():
                    if tenant_id and artifact.get('tenant_id') != tenant_id:
                        continue
                    if owner_id and artifact.get('owner_id') != owner_id:
                        continue
                    if artifact_type and artifact.get('artifact_type') != artifact_type:
                        continue
                    if security_level and artifact.get('security_level') != security_level:
                        continue
                    if status and artifact.get('status') != status:
                        continue
                    artifacts.append(artifact)
                return artifacts[offset:offset + limit]
            
            from backend.schemas.artifact_models import ArtifactModel
            with self._db_manager.get_session() as session:
                query = session.query(ArtifactModel)
                if tenant_id:
                    query = query.filter(ArtifactModel.tenant_id == tenant_id)
                if owner_id:
                    query = query.filter(ArtifactModel.owner_id == owner_id)
                if artifact_type:
                    query = query.filter(ArtifactModel.artifact_type == artifact_type)
                if security_level:
                    query = query.filter(ArtifactModel.security_level == security_level)
                if status:
                    query = query.filter(ArtifactModel.status == status)
                
                query = query.order_by(ArtifactModel.created_at.desc())
                query = query.offset(offset).limit(limit)
                return [self._model_to_dict(m) for m in query.all()]
                
        except Exception as e:
            logger.error(f"Failed to get artifacts: {e}")
            return []
    
    def update(self, artifact_id: str, updates: Dict[str, Any], tenant_id: str = None) -> bool:
        """更新工件"""
        try:
            updates['updated_at'] = datetime.utcnow()
            
            if self._use_memory:
                if artifact_id in self._artifacts:
                    artifact = self._artifacts[artifact_id]
                    if tenant_id and artifact.get('tenant_id') != tenant_id:
                        return False
                    artifact.update(updates)
                    return True
                return False
            
            from backend.schemas.artifact_models import ArtifactModel
            with self._db_manager.get_session() as session:
                query = session.query(ArtifactModel).filter(
                    ArtifactModel.id == artifact_id
                )
                if tenant_id:
                    query = query.filter(ArtifactModel.tenant_id == tenant_id)
                result = query.update(updates)
                session.commit()
                return result > 0
                
        except Exception as e:
            logger.error(f"Failed to update artifact: {e}")
            return False
    
    def delete(self, artifact_id: str, tenant_id: str = None) -> bool:
        """删除工件"""
        try:
            if self._use_memory:
                if artifact_id in self._artifacts:
                    artifact = self._artifacts[artifact_id]
                    if tenant_id and artifact.get('tenant_id') != tenant_id:
                        return False
                    del self._artifacts[artifact_id]
                    return True
                return False
            
            from backend.schemas.artifact_models import ArtifactModel
            with self._db_manager.get_session() as session:
                query = session.query(ArtifactModel).filter(
                    ArtifactModel.id == artifact_id
                )
                if tenant_id:
                    query = query.filter(ArtifactModel.tenant_id == tenant_id)
                result = query.delete()
                session.commit()
                return result > 0
                
        except Exception as e:
            logger.error(f"Failed to delete artifact: {e}")
            return False
    
    def update_status(self, artifact_id: str, status: str, tenant_id: str = None) -> bool:
        """更新工件状态"""
        return self.update(artifact_id, {'status': status}, tenant_id)
    
    def increment_version_count(self, artifact_id: str, size_delta: int = 0) -> bool:
        """增加版本计数"""
        try:
            if self._use_memory:
                if artifact_id in self._artifacts:
                    artifact = self._artifacts[artifact_id]
                    artifact['version_count'] = artifact.get('version_count', 0) + 1
                    artifact['total_size'] = artifact.get('total_size', 0) + size_delta
                    artifact['updated_at'] = datetime.utcnow()
                    return True
                return False
            
            from backend.schemas.artifact_models import ArtifactModel
            from sqlalchemy import func
            with self._db_manager.get_session() as session:
                result = session.query(ArtifactModel).filter(
                    ArtifactModel.id == artifact_id
                ).update({
                    'version_count': ArtifactModel.version_count + 1,
                    'total_size': ArtifactModel.total_size + size_delta,
                    'updated_at': datetime.utcnow()
                })
                session.commit()
                return result > 0
                
        except Exception as e:
            logger.error(f"Failed to increment version count: {e}")
            return False
    
    def _model_to_dict(self, model) -> Dict:
        """模型转字典"""
        return {
            'id': model.id,
            'tenant_id': model.tenant_id,
            'name': model.name,
            'description': model.description,
            'artifact_type': model.artifact_type,
            'security_level': model.security_level,
            'status': model.status,
            'owner_id': model.owner_id,
            'current_version': model.current_version,
            'version_count': model.version_count,
            'total_size': model.total_size,
            'tags': model.tags or [],
            'metadata': model.metadata or {},
            'policy_id': model.policy_id,
            'created_at': model.created_at,
            'updated_at': model.updated_at
        }


# ==================== 工件版本仓库 ====================

class ArtifactVersionRepository(BaseArtifactRepository):
    """工件版本仓库"""
    
    # 内存存储
    _versions: Dict[str, Dict] = {}
    
    def create(self, version_data: Dict[str, Any]) -> Optional[Dict]:
        """创建版本"""
        try:
            version_id = version_data.get('id') or f"ver_{uuid.uuid4().hex[:12]}"
            now = datetime.utcnow()
            
            version = {
                'id': version_id,
                'artifact_id': version_data['artifact_id'],
                'version': version_data['version'],
                'status': version_data.get('status', 'active'),
                'file_path': version_data.get('file_path'),
                'file_size': version_data.get('file_size', 0),
                'file_hash': version_data.get('file_hash'),
                'mime_type': version_data.get('mime_type'),
                'changelog': version_data.get('changelog', ''),
                'tags': version_data.get('tags', []),
                'metadata': version_data.get('metadata', {}),
                'created_by': version_data['created_by'],
                'created_at': now
            }
            
            if self._use_memory:
                self._versions[version_id] = version
                return version
            
            from backend.schemas.artifact_models import ArtifactVersionModel
            with self._db_manager.get_session() as session:
                model = ArtifactVersionModel(**version)
                session.add(model)
                session.commit()
                return version
                
        except Exception as e:
            logger.error(f"Failed to create version: {e}")
            return None
    
    def get_by_id(self, version_id: str) -> Optional[Dict]:
        """获取版本"""
        try:
            if self._use_memory:
                return self._versions.get(version_id)
            
            from backend.schemas.artifact_models import ArtifactVersionModel
            with self._db_manager.get_session() as session:
                model = session.query(ArtifactVersionModel).filter(
                    ArtifactVersionModel.id == version_id
                ).first()
                if model:
                    return self._model_to_dict(model)
                return None
                
        except Exception as e:
            logger.error(f"Failed to get version: {e}")
            return None
    
    def get_by_artifact_and_version(self, artifact_id: str, version: str) -> Optional[Dict]:
        """根据工件ID和版本号获取版本"""
        try:
            if self._use_memory:
                for v in self._versions.values():
                    if v['artifact_id'] == artifact_id and v['version'] == version:
                        return v
                return None
            
            from backend.schemas.artifact_models import ArtifactVersionModel
            with self._db_manager.get_session() as session:
                model = session.query(ArtifactVersionModel).filter(
                    ArtifactVersionModel.artifact_id == artifact_id,
                    ArtifactVersionModel.version == version
                ).first()
                if model:
                    return self._model_to_dict(model)
                return None
                
        except Exception as e:
            logger.error(f"Failed to get version by artifact and version: {e}")
            return None
    
    def get_by_artifact(self, artifact_id: str, status: str = None) -> List[Dict]:
        """获取工件的所有版本"""
        try:
            if self._use_memory:
                versions = []
                for v in self._versions.values():
                    if v['artifact_id'] == artifact_id:
                        if status is None or v.get('status') == status:
                            versions.append(v)
                return sorted(versions, key=lambda x: x.get('created_at', datetime.min), reverse=True)
            
            from backend.schemas.artifact_models import ArtifactVersionModel
            with self._db_manager.get_session() as session:
                query = session.query(ArtifactVersionModel).filter(
                    ArtifactVersionModel.artifact_id == artifact_id
                )
                if status:
                    query = query.filter(ArtifactVersionModel.status == status)
                query = query.order_by(ArtifactVersionModel.created_at.desc())
                return [self._model_to_dict(m) for m in query.all()]
                
        except Exception as e:
            logger.error(f"Failed to get versions: {e}")
            return []
    
    def update_status(self, version_id: str, status: str) -> bool:
        """更新版本状态"""
        try:
            if self._use_memory:
                if version_id in self._versions:
                    self._versions[version_id]['status'] = status
                    return True
                return False
            
            from backend.schemas.artifact_models import ArtifactVersionModel
            with self._db_manager.get_session() as session:
                result = session.query(ArtifactVersionModel).filter(
                    ArtifactVersionModel.id == version_id
                ).update({'status': status})
                session.commit()
                return result > 0
                
        except Exception as e:
            logger.error(f"Failed to update version status: {e}")
            return False
    
    def delete(self, version_id: str) -> bool:
        """删除版本"""
        try:
            if self._use_memory:
                if version_id in self._versions:
                    del self._versions[version_id]
                    return True
                return False
            
            from backend.schemas.artifact_models import ArtifactVersionModel
            with self._db_manager.get_session() as session:
                result = session.query(ArtifactVersionModel).filter(
                    ArtifactVersionModel.id == version_id
                ).delete()
                session.commit()
                return result > 0
                
        except Exception as e:
            logger.error(f"Failed to delete version: {e}")
            return False
    
    def cleanup_old_versions(self, artifact_id: str, keep_count: int = 10) -> int:
        """清理旧版本，只保留最新的N个"""
        try:
            versions = self.get_by_artifact(artifact_id)
            if len(versions) <= keep_count:
                return 0
            
            # 保留最新的N个版本，删除其余的
            to_delete = versions[keep_count:]
            deleted = 0
            for v in to_delete:
                if self.delete(v['id']):
                    deleted += 1
            return deleted
            
        except Exception as e:
            logger.error(f"Failed to cleanup old versions: {e}")
            return 0
    
    def _model_to_dict(self, model) -> Dict:
        """模型转字典"""
        return {
            'id': model.id,
            'artifact_id': model.artifact_id,
            'version': model.version,
            'status': model.status,
            'file_path': model.file_path,
            'file_size': model.file_size,
            'file_hash': model.file_hash,
            'mime_type': model.mime_type,
            'changelog': model.changelog,
            'tags': model.tags or [],
            'metadata': model.metadata or {},
            'created_by': model.created_by,
            'created_at': model.created_at
        }


# ==================== 文件元数据仓库 ====================

class FileMetadataRepository(BaseArtifactRepository):
    """文件元数据仓库"""
    
    # 内存存储
    _files: Dict[str, Dict] = {}
    
    def create(self, file_data: Dict[str, Any]) -> Optional[Dict]:
        """创建文件元数据"""
        try:
            file_id = file_data.get('id') or f"file_{uuid.uuid4().hex[:12]}"
            now = datetime.utcnow()
            
            file_meta = {
                'id': file_id,
                'tenant_id': file_data.get('tenant_id'),
                'original_name': file_data['original_name'],
                'stored_name': file_data['stored_name'],
                'file_path': file_data['file_path'],
                'file_type': file_data['file_type'],
                'mime_type': file_data.get('mime_type'),
                'size': file_data['size'],
                'hash_sha256': file_data.get('hash_sha256'),
                'hash_md5': file_data.get('hash_md5'),
                'security_level': file_data['security_level'],
                'artifact_type': file_data.get('artifact_type'),
                'artifact_id': file_data.get('artifact_id'),
                'version_id': file_data.get('version_id'),
                'owner_id': file_data['owner_id'],
                'is_encrypted': file_data.get('is_encrypted', False),
                'encryption_key_id': file_data.get('encryption_key_id'),
                'tags': file_data.get('tags', []),
                'metadata': file_data.get('metadata', {}),
                'created_at': now,
                'updated_at': now,
                'accessed_at': now,
                'expires_at': file_data.get('expires_at')
            }
            
            if self._use_memory:
                self._files[file_id] = file_meta
                return file_meta
            
            from backend.schemas.artifact_models import FileMetadataModel
            with self._db_manager.get_session() as session:
                model = FileMetadataModel(**file_meta)
                session.add(model)
                session.commit()
                return file_meta
                
        except Exception as e:
            logger.error(f"Failed to create file metadata: {e}")
            return None
    
    def get_by_id(self, file_id: str, tenant_id: str = None) -> Optional[Dict]:
        """获取文件元数据"""
        try:
            if self._use_memory:
                file_meta = self._files.get(file_id)
                if file_meta and (tenant_id is None or file_meta.get('tenant_id') == tenant_id):
                    return file_meta
                return None
            
            from backend.schemas.artifact_models import FileMetadataModel
            with self._db_manager.get_session() as session:
                query = session.query(FileMetadataModel).filter(
                    FileMetadataModel.id == file_id
                )
                if tenant_id:
                    query = query.filter(FileMetadataModel.tenant_id == tenant_id)
                model = query.first()
                if model:
                    return self._model_to_dict(model)
                return None
                
        except Exception as e:
            logger.error(f"Failed to get file metadata: {e}")
            return None
    
    def get_by_hash(self, hash_sha256: str, tenant_id: str = None) -> Optional[Dict]:
        """根据哈希获取文件"""
        try:
            if self._use_memory:
                for f in self._files.values():
                    if f.get('hash_sha256') == hash_sha256:
                        if tenant_id is None or f.get('tenant_id') == tenant_id:
                            return f
                return None
            
            from backend.schemas.artifact_models import FileMetadataModel
            with self._db_manager.get_session() as session:
                query = session.query(FileMetadataModel).filter(
                    FileMetadataModel.hash_sha256 == hash_sha256
                )
                if tenant_id:
                    query = query.filter(FileMetadataModel.tenant_id == tenant_id)
                model = query.first()
                if model:
                    return self._model_to_dict(model)
                return None
                
        except Exception as e:
            logger.error(f"Failed to get file by hash: {e}")
            return None
    
    def get_by_owner(
        self,
        owner_id: str,
        tenant_id: str = None,
        artifact_type: str = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict]:
        """获取用户的文件列表"""
        try:
            if self._use_memory:
                files = []
                for f in self._files.values():
                    if f.get('owner_id') != owner_id:
                        continue
                    if tenant_id and f.get('tenant_id') != tenant_id:
                        continue
                    if artifact_type and f.get('artifact_type') != artifact_type:
                        continue
                    files.append(f)
                return files[offset:offset + limit]
            
            from backend.schemas.artifact_models import FileMetadataModel
            with self._db_manager.get_session() as session:
                query = session.query(FileMetadataModel).filter(
                    FileMetadataModel.owner_id == owner_id
                )
                if tenant_id:
                    query = query.filter(FileMetadataModel.tenant_id == tenant_id)
                if artifact_type:
                    query = query.filter(FileMetadataModel.artifact_type == artifact_type)
                query = query.order_by(FileMetadataModel.created_at.desc())
                query = query.offset(offset).limit(limit)
                return [self._model_to_dict(m) for m in query.all()]
                
        except Exception as e:
            logger.error(f"Failed to get files by owner: {e}")
            return []
    
    def update_access_time(self, file_id: str) -> bool:
        """更新访问时间"""
        try:
            now = datetime.utcnow()
            
            if self._use_memory:
                if file_id in self._files:
                    self._files[file_id]['accessed_at'] = now
                    return True
                return False
            
            from backend.schemas.artifact_models import FileMetadataModel
            with self._db_manager.get_session() as session:
                result = session.query(FileMetadataModel).filter(
                    FileMetadataModel.id == file_id
                ).update({'accessed_at': now})
                session.commit()
                return result > 0
                
        except Exception as e:
            logger.error(f"Failed to update access time: {e}")
            return False
    
    def delete(self, file_id: str, tenant_id: str = None) -> bool:
        """删除文件元数据"""
        try:
            if self._use_memory:
                if file_id in self._files:
                    f = self._files[file_id]
                    if tenant_id and f.get('tenant_id') != tenant_id:
                        return False
                    del self._files[file_id]
                    return True
                return False
            
            from backend.schemas.artifact_models import FileMetadataModel
            with self._db_manager.get_session() as session:
                query = session.query(FileMetadataModel).filter(
                    FileMetadataModel.id == file_id
                )
                if tenant_id:
                    query = query.filter(FileMetadataModel.tenant_id == tenant_id)
                result = query.delete()
                session.commit()
                return result > 0
                
        except Exception as e:
            logger.error(f"Failed to delete file metadata: {e}")
            return False
    
    def get_expired_files(self, before_date: datetime = None) -> List[Dict]:
        """获取过期文件"""
        try:
            if before_date is None:
                before_date = datetime.utcnow()
            
            if self._use_memory:
                expired = []
                for f in self._files.values():
                    expires_at = f.get('expires_at')
                    if expires_at and expires_at < before_date:
                        expired.append(f)
                return expired
            
            from backend.schemas.artifact_models import FileMetadataModel
            with self._db_manager.get_session() as session:
                query = session.query(FileMetadataModel).filter(
                    FileMetadataModel.expires_at.isnot(None),
                    FileMetadataModel.expires_at < before_date
                )
                return [self._model_to_dict(m) for m in query.all()]
                
        except Exception as e:
            logger.error(f"Failed to get expired files: {e}")
            return []
    
    def _model_to_dict(self, model) -> Dict:
        """模型转字典"""
        return {
            'id': model.id,
            'tenant_id': model.tenant_id,
            'original_name': model.original_name,
            'stored_name': model.stored_name,
            'file_path': model.file_path,
            'file_type': model.file_type,
            'mime_type': model.mime_type,
            'size': model.size,
            'hash_sha256': model.hash_sha256,
            'hash_md5': model.hash_md5,
            'security_level': model.security_level,
            'artifact_type': model.artifact_type,
            'artifact_id': model.artifact_id,
            'version_id': model.version_id,
            'owner_id': model.owner_id,
            'is_encrypted': model.is_encrypted,
            'encryption_key_id': model.encryption_key_id,
            'tags': model.tags or [],
            'metadata': model.metadata or {},
            'created_at': model.created_at,
            'updated_at': model.updated_at,
            'accessed_at': model.accessed_at,
            'expires_at': model.expires_at
        }


# ==================== 依赖关系仓库 ====================

class ArtifactDependencyRepository(BaseArtifactRepository):
    """工件依赖关系仓库"""
    
    # 内存存储
    _dependencies: Dict[str, Dict] = {}
    
    def create(self, dep_data: Dict[str, Any]) -> Optional[Dict]:
        """创建依赖关系"""
        try:
            dep_id = dep_data.get('id') or f"dep_{uuid.uuid4().hex[:12]}"
            now = datetime.utcnow()
            
            dependency = {
                'id': dep_id,
                'source_artifact_id': dep_data['source_artifact_id'],
                'target_artifact_id': dep_data['target_artifact_id'],
                'dependency_type': dep_data.get('dependency_type', 'required'),
                'version_constraint': dep_data.get('version_constraint', '*'),
                'created_by': dep_data.get('created_by'),
                'created_at': now
            }
            
            if self._use_memory:
                self._dependencies[dep_id] = dependency
                return dependency
            
            from backend.schemas.artifact_models import ArtifactDependencyModel
            with self._db_manager.get_session() as session:
                model = ArtifactDependencyModel(**dependency)
                session.add(model)
                session.commit()
                return dependency
                
        except Exception as e:
            logger.error(f"Failed to create dependency: {e}")
            return None
    
    def get_dependencies(self, artifact_id: str) -> List[Dict]:
        """获取工件的依赖列表"""
        try:
            if self._use_memory:
                deps = []
                for d in self._dependencies.values():
                    if d['source_artifact_id'] == artifact_id:
                        deps.append(d)
                return deps
            
            from backend.schemas.artifact_models import ArtifactDependencyModel
            with self._db_manager.get_session() as session:
                models = session.query(ArtifactDependencyModel).filter(
                    ArtifactDependencyModel.source_artifact_id == artifact_id
                ).all()
                return [self._model_to_dict(m) for m in models]
                
        except Exception as e:
            logger.error(f"Failed to get dependencies: {e}")
            return []
    
    def get_dependents(self, artifact_id: str) -> List[Dict]:
        """获取依赖此工件的列表"""
        try:
            if self._use_memory:
                deps = []
                for d in self._dependencies.values():
                    if d['target_artifact_id'] == artifact_id:
                        deps.append(d)
                return deps
            
            from backend.schemas.artifact_models import ArtifactDependencyModel
            with self._db_manager.get_session() as session:
                models = session.query(ArtifactDependencyModel).filter(
                    ArtifactDependencyModel.target_artifact_id == artifact_id
                ).all()
                return [self._model_to_dict(m) for m in models]
                
        except Exception as e:
            logger.error(f"Failed to get dependents: {e}")
            return []
    
    def delete(self, source_id: str, target_id: str) -> bool:
        """删除依赖关系"""
        try:
            if self._use_memory:
                to_delete = None
                for dep_id, d in self._dependencies.items():
                    if d['source_artifact_id'] == source_id and d['target_artifact_id'] == target_id:
                        to_delete = dep_id
                        break
                if to_delete:
                    del self._dependencies[to_delete]
                    return True
                return False
            
            from backend.schemas.artifact_models import ArtifactDependencyModel
            with self._db_manager.get_session() as session:
                result = session.query(ArtifactDependencyModel).filter(
                    ArtifactDependencyModel.source_artifact_id == source_id,
                    ArtifactDependencyModel.target_artifact_id == target_id
                ).delete()
                session.commit()
                return result > 0
                
        except Exception as e:
            logger.error(f"Failed to delete dependency: {e}")
            return False
    
    def _model_to_dict(self, model) -> Dict:
        """模型转字典"""
        return {
            'id': model.id,
            'source_artifact_id': model.source_artifact_id,
            'target_artifact_id': model.target_artifact_id,
            'dependency_type': model.dependency_type,
            'version_constraint': model.version_constraint,
            'created_by': model.created_by,
            'created_at': model.created_at
        }


# ==================== 访问日志仓库 ====================

class ArtifactAccessLogRepository(BaseArtifactRepository):
    """工件访问日志仓库"""
    
    # 内存存储
    _logs: List[Dict] = []
    
    def create(self, log_data: Dict[str, Any]) -> Optional[Dict]:
        """创建访问日志"""
        try:
            log_id = log_data.get('id') or f"log_{uuid.uuid4().hex[:12]}"
            now = datetime.utcnow()
            
            log = {
                'id': log_id,
                'tenant_id': log_data.get('tenant_id'),
                'artifact_id': log_data.get('artifact_id'),
                'file_id': log_data.get('file_id'),
                'user_id': log_data['user_id'],
                'operation': log_data['operation'],
                'result': log_data['result'],
                'ip_address': log_data.get('ip_address'),
                'user_agent': log_data.get('user_agent'),
                'details': log_data.get('details', {}),
                'created_at': now
            }
            
            if self._use_memory:
                self._logs.append(log)
                # 保留最近10000条日志
                if len(self._logs) > 10000:
                    self._logs = self._logs[-10000:]
                return log
            
            from backend.schemas.artifact_models import ArtifactAccessLogModel
            with self._db_manager.get_session() as session:
                model = ArtifactAccessLogModel(**log)
                session.add(model)
                session.commit()
                return log
                
        except Exception as e:
            logger.error(f"Failed to create access log: {e}")
            return None
    
    def query(
        self,
        tenant_id: str = None,
        artifact_id: str = None,
        user_id: str = None,
        operation: str = None,
        start_time: datetime = None,
        end_time: datetime = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict]:
        """查询访问日志"""
        try:
            if self._use_memory:
                logs = []
                for log in self._logs:
                    if tenant_id and log.get('tenant_id') != tenant_id:
                        continue
                    if artifact_id and log.get('artifact_id') != artifact_id:
                        continue
                    if user_id and log.get('user_id') != user_id:
                        continue
                    if operation and log.get('operation') != operation:
                        continue
                    if start_time and log.get('created_at', datetime.min) < start_time:
                        continue
                    if end_time and log.get('created_at', datetime.max) > end_time:
                        continue
                    logs.append(log)
                # 按时间倒序
                logs.sort(key=lambda x: x.get('created_at', datetime.min), reverse=True)
                return logs[offset:offset + limit]
            
            from backend.schemas.artifact_models import ArtifactAccessLogModel
            with self._db_manager.get_session() as session:
                query = session.query(ArtifactAccessLogModel)
                if tenant_id:
                    query = query.filter(ArtifactAccessLogModel.tenant_id == tenant_id)
                if artifact_id:
                    query = query.filter(ArtifactAccessLogModel.artifact_id == artifact_id)
                if user_id:
                    query = query.filter(ArtifactAccessLogModel.user_id == user_id)
                if operation:
                    query = query.filter(ArtifactAccessLogModel.operation == operation)
                if start_time:
                    query = query.filter(ArtifactAccessLogModel.created_at >= start_time)
                if end_time:
                    query = query.filter(ArtifactAccessLogModel.created_at <= end_time)
                
                query = query.order_by(ArtifactAccessLogModel.created_at.desc())
                query = query.offset(offset).limit(limit)
                return [self._model_to_dict(m) for m in query.all()]
                
        except Exception as e:
            logger.error(f"Failed to query access logs: {e}")
            return []
    
    def _model_to_dict(self, model) -> Dict:
        """模型转字典"""
        return {
            'id': model.id,
            'tenant_id': model.tenant_id,
            'artifact_id': model.artifact_id,
            'file_id': model.file_id,
            'user_id': model.user_id,
            'operation': model.operation,
            'result': model.result,
            'ip_address': model.ip_address,
            'user_agent': model.user_agent,
            'details': model.details or {},
            'created_at': model.created_at
        }


# ==================== 单例获取函数 ====================

_policy_repo = None
_artifact_repo = None
_version_repo = None
_file_repo = None
_dependency_repo = None
_access_log_repo = None


def get_security_policy_repository(use_memory: bool = False) -> SecurityPolicyRepository:
    """获取安全策略仓库实例"""
    global _policy_repo
    if _policy_repo is None:
        _policy_repo = SecurityPolicyRepository(use_memory=use_memory)
    return _policy_repo


def get_artifact_repository(use_memory: bool = False) -> ArtifactRepository:
    """获取工件仓库实例"""
    global _artifact_repo
    if _artifact_repo is None:
        _artifact_repo = ArtifactRepository(use_memory=use_memory)
    return _artifact_repo


def get_artifact_version_repository(use_memory: bool = False) -> ArtifactVersionRepository:
    """获取工件版本仓库实例"""
    global _version_repo
    if _version_repo is None:
        _version_repo = ArtifactVersionRepository(use_memory=use_memory)
    return _version_repo


def get_file_metadata_repository(use_memory: bool = False) -> FileMetadataRepository:
    """获取文件元数据仓库实例"""
    global _file_repo
    if _file_repo is None:
        _file_repo = FileMetadataRepository(use_memory=use_memory)
    return _file_repo


def get_artifact_dependency_repository(use_memory: bool = False) -> ArtifactDependencyRepository:
    """获取依赖关系仓库实例"""
    global _dependency_repo
    if _dependency_repo is None:
        _dependency_repo = ArtifactDependencyRepository(use_memory=use_memory)
    return _dependency_repo


def get_artifact_access_log_repository(use_memory: bool = False) -> ArtifactAccessLogRepository:
    """获取访问日志仓库实例"""
    global _access_log_repo
    if _access_log_repo is None:
        _access_log_repo = ArtifactAccessLogRepository(use_memory=use_memory)
    return _access_log_repo


def reset_artifact_repositories():
    """重置所有仓库实例（用于测试）"""
    global _policy_repo, _artifact_repo, _version_repo, _file_repo, _dependency_repo, _access_log_repo
    _policy_repo = None
    _artifact_repo = None
    _version_repo = None
    _file_repo = None
    _dependency_repo = None
    _access_log_repo = None

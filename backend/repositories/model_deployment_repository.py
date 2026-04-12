"""模型部署数据访问层

提供模型部署相关的数据库访问功能，包括：
- 模型部署记录 (ModelDeployment)
- 部署日志 (ModelDeploymentLog)
- 模型服务 (ModelService)
"""

import json
import logging
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime
import uuid

from backend.core.exceptions import ValidationError, DatabaseError
from backend.schemas.training_models import (
    ModelDeployment, ModelDeploymentLog, ModelService
)

logger = logging.getLogger(__name__)


def _generate_id() -> str:
    """生成唯一ID"""
    return str(uuid.uuid4())


class ModelDeploymentRepository:
    """模型部署数据访问层"""
    
    def __init__(self, use_memory_storage: bool = False):
        """初始化模型部署仓库
        
        Args:
            use_memory_storage: 是否使用内存存储
        """
        self._use_memory_storage = use_memory_storage
        
        if use_memory_storage:
            self._deployments: Dict[str, Dict] = {}
            self._db_manager = None
        else:
            try:
                from backend.modules.database.manager import get_database_manager
                self._db_manager = get_database_manager()
            except ImportError:
                logger.warning("Cannot import database service, falling back to memory storage")
                self._use_memory_storage = True
                self._deployments: Dict[str, Dict] = {}
                self._db_manager = None
    
    def create(self, deployment_data: Dict[str, Any]) -> ModelDeployment:
        """创建部署记录"""
        try:
            record_id = deployment_data.get('id') or _generate_id()
            deployment_id = deployment_data.get('deployment_id') or f"deploy_{_generate_id()[:8]}"
            
            if self._use_memory_storage:
                deployment_data['id'] = record_id
                deployment_data['deployment_id'] = deployment_id
                deployment_data['created_at'] = datetime.utcnow().isoformat()
                deployment_data['updated_at'] = datetime.utcnow().isoformat()
                deployment_data.setdefault('status', 'pending')
                deployment_data.setdefault('replicas', 1)
                deployment_data.setdefault('request_count', 0)
                deployment_data.setdefault('error_count', 0)
                self._deployments[record_id] = deployment_data
                return deployment_data
            
            with self._db_manager.get_db_session() as db:
                # 处理JSON字段
                resources = deployment_data.get('resources', {})
                if isinstance(resources, dict):
                    resources = json.dumps(resources)
                
                environment = deployment_data.get('environment', {})
                if isinstance(environment, dict):
                    environment = json.dumps(environment)
                
                tags = deployment_data.get('tags', [])
                if isinstance(tags, list):
                    tags = json.dumps(tags)
                
                extra_data = deployment_data.get('extra_data', {})
                if isinstance(extra_data, dict):
                    extra_data = json.dumps(extra_data)
                
                error_details = deployment_data.get('error_details', {})
                if isinstance(error_details, dict):
                    error_details = json.dumps(error_details)
                
                deployment = ModelDeployment(
                    id=record_id,
                    tenant_id=deployment_data.get('tenant_id'),
                    user_id=deployment_data['user_id'],
                    deployment_id=deployment_id,
                    model_id=deployment_data['model_id'],
                    mode=deployment_data.get('mode', 'online'),
                    release_strategy=deployment_data.get('release_strategy', 'rolling'),
                    status=deployment_data.get('status', 'pending'),
                    replicas=deployment_data.get('replicas', 1),
                    resources=resources,
                    environment=environment,
                    endpoint_url=deployment_data.get('endpoint_url'),
                    health_check_url=deployment_data.get('health_check_url'),
                    autoscaling=deployment_data.get('autoscaling', False),
                    health_check=deployment_data.get('health_check', True),
                    monitoring=deployment_data.get('monitoring', True),
                    canary_percent=deployment_data.get('canary_percent', 10),
                    rolling_step=deployment_data.get('rolling_step', 1),
                    ab_percent=deployment_data.get('ab_percent', 50),
                    version=deployment_data.get('version'),
                    tags=tags,
                    extra_data=extra_data,
                    error_details=error_details
                )
                db.add(deployment)
                db.commit()
                db.refresh(deployment)
                return deployment
                
        except Exception as e:
            logger.error(f"Failed to create deployment: {e}")
            raise DatabaseError(f"Failed to create deployment: {e}", operation="create_deployment")
    
    def get_by_id(self, record_id: str) -> Optional[ModelDeployment]:
        """根据ID获取部署记录"""
        try:
            if self._use_memory_storage:
                return self._deployments.get(record_id)
            
            with self._db_manager.get_db_session() as db:
                return db.query(ModelDeployment).filter(ModelDeployment.id == record_id).first()
                
        except Exception as e:
            logger.error(f"Failed to get deployment by id: {e}")
            return None
    
    def get_by_deployment_id(self, deployment_id: str) -> Optional[ModelDeployment]:
        """根据部署ID获取部署记录"""
        try:
            if self._use_memory_storage:
                for dep in self._deployments.values():
                    if dep.get('deployment_id') == deployment_id:
                        return dep
                return None
            
            with self._db_manager.get_db_session() as db:
                return db.query(ModelDeployment).filter(
                    ModelDeployment.deployment_id == deployment_id
                ).first()
                
        except Exception as e:
            logger.error(f"Failed to get deployment by deployment_id: {e}")
            return None
    
    def get_by_tenant(self, tenant_id: str, model_id: Optional[str] = None,
                     status: Optional[str] = None, mode: Optional[str] = None,
                     limit: int = 100, offset: int = 0) -> Tuple[List[ModelDeployment], int]:
        """获取租户的部署记录列表"""
        try:
            if self._use_memory_storage:
                deployments = [d for d in self._deployments.values() if d.get('tenant_id') == tenant_id]
                if model_id:
                    deployments = [d for d in deployments if d.get('model_id') == model_id]
                if status:
                    deployments = [d for d in deployments if d.get('status') == status]
                if mode:
                    deployments = [d for d in deployments if d.get('mode') == mode]
                
                # 按创建时间排序
                deployments.sort(key=lambda x: x.get('created_at', ''), reverse=True)
                total = len(deployments)
                return deployments[offset:offset + limit], total
            
            with self._db_manager.get_db_session() as db:
                query = db.query(ModelDeployment).filter(ModelDeployment.tenant_id == tenant_id)
                
                if model_id:
                    query = query.filter(ModelDeployment.model_id == model_id)
                if status:
                    query = query.filter(ModelDeployment.status == status)
                if mode:
                    query = query.filter(ModelDeployment.mode == mode)
                
                total = query.count()
                deployments = query.order_by(ModelDeployment.created_at.desc()).offset(offset).limit(limit).all()
                return deployments, total
                
        except Exception as e:
            logger.error(f"Failed to get deployments by tenant: {e}")
            return [], 0
    
    def get_by_user(self, user_id: str, tenant_id: Optional[str] = None,
                   limit: int = 100, offset: int = 0) -> Tuple[List[ModelDeployment], int]:
        """获取用户的部署记录"""
        try:
            if self._use_memory_storage:
                deployments = [d for d in self._deployments.values() if d.get('user_id') == user_id]
                if tenant_id:
                    deployments = [d for d in deployments if d.get('tenant_id') == tenant_id]
                deployments.sort(key=lambda x: x.get('created_at', ''), reverse=True)
                total = len(deployments)
                return deployments[offset:offset + limit], total
            
            with self._db_manager.get_db_session() as db:
                query = db.query(ModelDeployment).filter(ModelDeployment.user_id == user_id)
                if tenant_id:
                    query = query.filter(ModelDeployment.tenant_id == tenant_id)
                
                total = query.count()
                deployments = query.order_by(ModelDeployment.created_at.desc()).offset(offset).limit(limit).all()
                return deployments, total
                
        except Exception as e:
            logger.error(f"Failed to get deployments by user: {e}")
            return [], 0
    
    def get_by_model(self, model_id: str, tenant_id: Optional[str] = None,
                    limit: int = 100, offset: int = 0) -> Tuple[List[ModelDeployment], int]:
        """获取模型的部署记录"""
        try:
            if self._use_memory_storage:
                deployments = [d for d in self._deployments.values() if d.get('model_id') == model_id]
                if tenant_id:
                    deployments = [d for d in deployments if d.get('tenant_id') == tenant_id]
                deployments.sort(key=lambda x: x.get('created_at', ''), reverse=True)
                total = len(deployments)
                return deployments[offset:offset + limit], total
            
            with self._db_manager.get_db_session() as db:
                query = db.query(ModelDeployment).filter(ModelDeployment.model_id == model_id)
                if tenant_id:
                    query = query.filter(ModelDeployment.tenant_id == tenant_id)
                
                total = query.count()
                deployments = query.order_by(ModelDeployment.created_at.desc()).offset(offset).limit(limit).all()
                return deployments, total
                
        except Exception as e:
            logger.error(f"Failed to get deployments by model: {e}")
            return [], 0
    
    def get_running_deployments(self, tenant_id: Optional[str] = None) -> List[ModelDeployment]:
        """获取正在运行的部署"""
        try:
            running_statuses = ['pending', 'deploying', 'running']
            
            if self._use_memory_storage:
                deployments = [d for d in self._deployments.values() if d.get('status') in running_statuses]
                if tenant_id:
                    deployments = [d for d in deployments if d.get('tenant_id') == tenant_id]
                return deployments
            
            with self._db_manager.get_db_session() as db:
                query = db.query(ModelDeployment).filter(
                    ModelDeployment.status.in_(running_statuses)
                )
                if tenant_id:
                    query = query.filter(ModelDeployment.tenant_id == tenant_id)
                return query.all()
                
        except Exception as e:
            logger.error(f"Failed to get running deployments: {e}")
            return []
    
    def update(self, record_id: str, update_data: Dict[str, Any]) -> Optional[ModelDeployment]:
        """更新部署记录"""
        try:
            if self._use_memory_storage:
                if record_id in self._deployments:
                    self._deployments[record_id].update(update_data)
                    self._deployments[record_id]['updated_at'] = datetime.utcnow().isoformat()
                    return self._deployments[record_id]
                return None
            
            with self._db_manager.get_db_session() as db:
                deployment = db.query(ModelDeployment).filter(ModelDeployment.id == record_id).first()
                if not deployment:
                    return None
                
                for key, value in update_data.items():
                    if key in ('resources', 'environment', 'tags', 'extra_data', 'error_details') and isinstance(value, (dict, list)):
                        value = json.dumps(value)
                    if hasattr(deployment, key):
                        setattr(deployment, key, value)
                
                deployment.updated_at = datetime.utcnow()
                db.commit()
                db.refresh(deployment)
                return deployment
                
        except Exception as e:
            logger.error(f"Failed to update deployment: {e}")
            raise DatabaseError(f"Failed to update deployment: {e}", operation="update_deployment")
    
    def update_by_deployment_id(self, deployment_id: str, update_data: Dict[str, Any]) -> Optional[ModelDeployment]:
        """根据部署ID更新记录"""
        try:
            if self._use_memory_storage:
                for record_id, dep in self._deployments.items():
                    if dep.get('deployment_id') == deployment_id:
                        return self.update(record_id, update_data)
                return None
            
            with self._db_manager.get_db_session() as db:
                deployment = db.query(ModelDeployment).filter(
                    ModelDeployment.deployment_id == deployment_id
                ).first()
                if not deployment:
                    return None
                
                for key, value in update_data.items():
                    if key in ('resources', 'environment', 'tags', 'extra_data', 'error_details') and isinstance(value, (dict, list)):
                        value = json.dumps(value)
                    if hasattr(deployment, key):
                        setattr(deployment, key, value)
                
                deployment.updated_at = datetime.utcnow()
                db.commit()
                db.refresh(deployment)
                return deployment
                
        except Exception as e:
            logger.error(f"Failed to update deployment by deployment_id: {e}")
            raise DatabaseError(f"Failed to update deployment: {e}", operation="update_deployment")
    
    def update_status(self, deployment_id: str, status: str,
                     error_message: Optional[str] = None) -> Optional[ModelDeployment]:
        """更新部署状态"""
        update_data = {'status': status}
        
        if status in ('running',):
            update_data['started_at'] = datetime.utcnow()
        elif status in ('stopped', 'failed', 'rolled_back'):
            update_data['stopped_at'] = datetime.utcnow()
        
        if error_message:
            update_data['error_message'] = error_message
        
        return self.update_by_deployment_id(deployment_id, update_data)
    
    def get_deployment_versions(self, model_id: str, tenant_id: Optional[str] = None,
                                limit: int = 10) -> List[Dict[str, Any]]:
        """获取模型的部署版本历史
        
        Args:
            model_id: 模型ID
            tenant_id: 租户ID
            limit: 返回的最大版本数
            
        Returns:
            版本历史列表，包含版本号、状态、部署时间等信息
        """
        try:
            if self._use_memory_storage:
                deployments = [
                    d for d in self._deployments.values()
                    if d.get('model_id') == model_id and d.get('version')
                ]
                if tenant_id:
                    deployments = [d for d in deployments if d.get('tenant_id') == tenant_id]
                
                # 按创建时间倒序排列
                deployments.sort(key=lambda x: x.get('created_at', ''), reverse=True)
                
                versions = []
                for dep in deployments[:limit]:
                    versions.append({
                        'deployment_id': dep.get('deployment_id'),
                        'version': dep.get('version'),
                        'previous_version': dep.get('previous_version'),
                        'status': dep.get('status'),
                        'mode': dep.get('mode'),
                        'replicas': dep.get('replicas'),
                        'endpoint_url': dep.get('endpoint_url'),
                        'created_at': dep.get('created_at'),
                        'started_at': dep.get('started_at'),
                        'stopped_at': dep.get('stopped_at'),
                        'is_current': dep.get('status') == 'running'
                    })
                return versions
            
            with self._db_manager.get_db_session() as db:
                query = db.query(ModelDeployment).filter(
                    ModelDeployment.model_id == model_id,
                    ModelDeployment.version != None
                )
                if tenant_id:
                    query = query.filter(ModelDeployment.tenant_id == tenant_id)
                
                deployments = query.order_by(ModelDeployment.created_at.desc()).limit(limit).all()
                
                versions = []
                for dep in deployments:
                    versions.append({
                        'deployment_id': dep.deployment_id,
                        'version': dep.version,
                        'previous_version': dep.previous_version,
                        'status': dep.status,
                        'mode': dep.mode,
                        'replicas': dep.replicas,
                        'endpoint_url': dep.endpoint_url,
                        'created_at': dep.created_at.isoformat() if dep.created_at else None,
                        'started_at': dep.started_at.isoformat() if dep.started_at else None,
                        'stopped_at': dep.stopped_at.isoformat() if dep.stopped_at else None,
                        'is_current': dep.status == 'running'
                    })
                return versions
                
        except Exception as e:
            logger.error(f"Failed to get deployment versions: {e}")
            return []
    
    def get_rollback_target(self, deployment_id: str, target_version: Optional[str] = None,
                           tenant_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """获取回滚目标版本的信息
        
        Args:
            deployment_id: 当前部署ID
            target_version: 目标版本号，如果为None则回滚到上一版本
            tenant_id: 租户ID
            
        Returns:
            目标版本的部署信息
        """
        try:
            # 先获取当前部署
            current = self.get_by_deployment_id(deployment_id)
            if not current:
                logger.warning(f"Deployment not found: {deployment_id}")
                return None
            
            if self._use_memory_storage:
                current_data = current if isinstance(current, dict) else current
                model_id = current_data.get('model_id')
                current_version = current_data.get('version')
                prev_version = current_data.get('previous_version')
                
                # 确定目标版本
                rollback_version = target_version or prev_version
                if not rollback_version:
                    logger.warning(f"No target version for rollback: {deployment_id}")
                    return None
                
                # 查找目标版本的部署记录
                for dep in self._deployments.values():
                    if (dep.get('model_id') == model_id and 
                        dep.get('version') == rollback_version):
                        if tenant_id and dep.get('tenant_id') != tenant_id:
                            continue
                        return {
                            'source_deployment_id': deployment_id,
                            'source_version': current_version,
                            'target_deployment_id': dep.get('deployment_id'),
                            'target_version': rollback_version,
                            'target_config': {
                                'mode': dep.get('mode'),
                                'replicas': dep.get('replicas'),
                                'resources': dep.get('resources'),
                                'environment': dep.get('environment'),
                                'endpoint_url': dep.get('endpoint_url'),
                                'release_strategy': dep.get('release_strategy'),
                            },
                            'model_id': model_id
                        }
                
                return None
            
            with self._db_manager.get_db_session() as db:
                current_dep = db.query(ModelDeployment).filter(
                    ModelDeployment.deployment_id == deployment_id
                ).first()
                
                if not current_dep:
                    return None
                
                model_id = current_dep.model_id
                current_version = current_dep.version
                prev_version = current_dep.previous_version
                
                # 确定目标版本
                rollback_version = target_version or prev_version
                if not rollback_version:
                    return None
                
                # 查找目标版本
                query = db.query(ModelDeployment).filter(
                    ModelDeployment.model_id == model_id,
                    ModelDeployment.version == rollback_version
                )
                if tenant_id:
                    query = query.filter(ModelDeployment.tenant_id == tenant_id)
                
                target_dep = query.first()
                
                if not target_dep:
                    return None
                
                return {
                    'source_deployment_id': deployment_id,
                    'source_version': current_version,
                    'target_deployment_id': target_dep.deployment_id,
                    'target_version': rollback_version,
                    'target_config': {
                        'mode': target_dep.mode,
                        'replicas': target_dep.replicas,
                        'resources': target_dep.resources,
                        'environment': target_dep.environment,
                        'endpoint_url': target_dep.endpoint_url,
                        'release_strategy': target_dep.release_strategy,
                    },
                    'model_id': model_id
                }
                
        except Exception as e:
            logger.error(f"Failed to get rollback target: {e}")
            return None
    
    def execute_rollback(self, deployment_id: str, target_version: str,
                        rollback_reason: Optional[str] = None,
                        user_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """执行回滚操作
        
        更新部署记录的版本信息和状态。
        
        Args:
            deployment_id: 部署ID
            target_version: 目标版本
            rollback_reason: 回滚原因
            user_id: 执行回滚的用户ID
            
        Returns:
            回滚结果信息
        """
        try:
            if self._use_memory_storage:
                for record_id, dep in self._deployments.items():
                    if dep.get('deployment_id') == deployment_id:
                        old_version = dep.get('version')
                        
                        # 更新部署记录
                        dep['previous_version'] = old_version
                        dep['version'] = target_version
                        dep['status'] = 'rolled_back'
                        dep['stopped_at'] = datetime.utcnow().isoformat()
                        dep['updated_at'] = datetime.utcnow().isoformat()
                        
                        # 记录回滚信息到 extra_data
                        extra_data = dep.get('extra_data', {}) or {}
                        if isinstance(extra_data, str):
                            extra_data = json.loads(extra_data)
                        
                        rollback_history = extra_data.get('rollback_history', [])
                        rollback_history.append({
                            'from_version': old_version,
                            'to_version': target_version,
                            'reason': rollback_reason,
                            'user_id': user_id,
                            'timestamp': datetime.utcnow().isoformat()
                        })
                        extra_data['rollback_history'] = rollback_history
                        extra_data['last_rollback'] = {
                            'from_version': old_version,
                            'to_version': target_version,
                            'reason': rollback_reason,
                            'user_id': user_id,
                            'timestamp': datetime.utcnow().isoformat()
                        }
                        dep['extra_data'] = extra_data
                        
                        return {
                            'success': True,
                            'deployment_id': deployment_id,
                            'old_version': old_version,
                            'new_version': target_version,
                            'status': 'rolled_back',
                            'rollback_reason': rollback_reason,
                            'timestamp': datetime.utcnow().isoformat()
                        }
                
                return None
            
            with self._db_manager.get_db_session() as db:
                deployment = db.query(ModelDeployment).filter(
                    ModelDeployment.deployment_id == deployment_id
                ).first()
                
                if not deployment:
                    return None
                
                old_version = deployment.version
                
                # 更新部署记录
                deployment.previous_version = old_version
                deployment.version = target_version
                deployment.status = 'rolled_back'
                deployment.stopped_at = datetime.utcnow()
                deployment.updated_at = datetime.utcnow()
                
                # 更新 extra_data
                extra_data = deployment.extra_data or {}
                if isinstance(extra_data, str):
                    extra_data = json.loads(extra_data)
                
                rollback_history = extra_data.get('rollback_history', [])
                rollback_history.append({
                    'from_version': old_version,
                    'to_version': target_version,
                    'reason': rollback_reason,
                    'user_id': user_id,
                    'timestamp': datetime.utcnow().isoformat()
                })
                extra_data['rollback_history'] = rollback_history
                extra_data['last_rollback'] = {
                    'from_version': old_version,
                    'to_version': target_version,
                    'reason': rollback_reason,
                    'user_id': user_id,
                    'timestamp': datetime.utcnow().isoformat()
                }
                deployment.extra_data = json.dumps(extra_data)
                
                db.commit()
                
                return {
                    'success': True,
                    'deployment_id': deployment_id,
                    'old_version': old_version,
                    'new_version': target_version,
                    'status': 'rolled_back',
                    'rollback_reason': rollback_reason,
                    'timestamp': datetime.utcnow().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Failed to execute rollback: {e}")
            return None
    
    def get_rollback_history(self, deployment_id: str) -> List[Dict[str, Any]]:
        """获取部署的回滚历史
        
        Args:
            deployment_id: 部署ID
            
        Returns:
            回滚历史列表
        """
        try:
            deployment = self.get_by_deployment_id(deployment_id)
            if not deployment:
                return []
            
            if self._use_memory_storage:
                extra_data = deployment.get('extra_data', {}) or {}
            else:
                extra_data = deployment.extra_data or {}
            
            if isinstance(extra_data, str):
                extra_data = json.loads(extra_data)
            
            return extra_data.get('rollback_history', [])
            
        except Exception as e:
            logger.error(f"Failed to get rollback history: {e}")
            return []
    
    def delete(self, record_id: str) -> bool:
        """删除部署记录"""
        try:
            if self._use_memory_storage:
                if record_id in self._deployments:
                    del self._deployments[record_id]
                    return True
                return False
            
            with self._db_manager.get_db_session() as db:
                deployment = db.query(ModelDeployment).filter(ModelDeployment.id == record_id).first()
                if not deployment:
                    return False
                
                db.delete(deployment)
                db.commit()
                return True
                
        except Exception as e:
            logger.error(f"Failed to delete deployment: {e}")
            return False
    
    def delete_by_deployment_id(self, deployment_id: str) -> bool:
        """根据部署ID删除记录"""
        try:
            if self._use_memory_storage:
                for record_id, dep in list(self._deployments.items()):
                    if dep.get('deployment_id') == deployment_id:
                        del self._deployments[record_id]
                        return True
                return False
            
            with self._db_manager.get_db_session() as db:
                deployment = db.query(ModelDeployment).filter(
                    ModelDeployment.deployment_id == deployment_id
                ).first()
                if not deployment:
                    return False
                
                db.delete(deployment)
                db.commit()
                return True
                
        except Exception as e:
            logger.error(f"Failed to delete deployment: {e}")
            return False
    
    def increment_request_count(self, deployment_id: str, error: bool = False) -> bool:
        """增加请求计数"""
        try:
            if self._use_memory_storage:
                for dep in self._deployments.values():
                    if dep.get('deployment_id') == deployment_id:
                        dep['request_count'] = dep.get('request_count', 0) + 1
                        if error:
                            dep['error_count'] = dep.get('error_count', 0) + 1
                        return True
                return False
            
            with self._db_manager.get_db_session() as db:
                deployment = db.query(ModelDeployment).filter(
                    ModelDeployment.deployment_id == deployment_id
                ).first()
                if not deployment:
                    return False
                
                deployment.request_count = (deployment.request_count or 0) + 1
                if error:
                    deployment.error_count = (deployment.error_count or 0) + 1
                db.commit()
                return True
                
        except Exception as e:
            logger.error(f"Failed to increment request count: {e}")
            return False
    
    def get_statistics(self, tenant_id: str) -> Dict[str, Any]:
        """获取部署统计信息"""
        try:
            if self._use_memory_storage:
                deployments = [d for d in self._deployments.values() if d.get('tenant_id') == tenant_id]
                
                status_counts = {}
                mode_counts = {}
                total_replicas = 0
                total_requests = 0
                total_errors = 0
                
                for d in deployments:
                    status = d.get('status', 'unknown')
                    status_counts[status] = status_counts.get(status, 0) + 1
                    
                    mode = d.get('mode', 'unknown')
                    mode_counts[mode] = mode_counts.get(mode, 0) + 1
                    
                    total_replicas += d.get('replicas', 0)
                    total_requests += d.get('request_count', 0)
                    total_errors += d.get('error_count', 0)
                
                running_count = status_counts.get('running', 0)
                
                return {
                    'total_deployments': len(deployments),
                    'running_deployments': running_count,
                    'by_status': status_counts,
                    'by_mode': mode_counts,
                    'total_replicas': total_replicas,
                    'total_requests': total_requests,
                    'total_errors': total_errors,
                    'error_rate': round(total_errors / total_requests * 100, 2) if total_requests > 0 else 0
                }
            
            with self._db_manager.get_db_session() as db:
                from sqlalchemy import func as sql_func
                
                # 总数
                total = db.query(sql_func.count(ModelDeployment.id)).filter(
                    ModelDeployment.tenant_id == tenant_id
                ).scalar() or 0
                
                # 运行中数量
                running = db.query(sql_func.count(ModelDeployment.id)).filter(
                    ModelDeployment.tenant_id == tenant_id,
                    ModelDeployment.status == 'running'
                ).scalar() or 0
                
                # 按状态统计
                status_results = db.query(
                    ModelDeployment.status,
                    sql_func.count(ModelDeployment.id)
                ).filter(ModelDeployment.tenant_id == tenant_id).group_by(ModelDeployment.status).all()
                status_counts = {r[0]: r[1] for r in status_results}
                
                # 按模式统计
                mode_results = db.query(
                    ModelDeployment.mode,
                    sql_func.count(ModelDeployment.id)
                ).filter(ModelDeployment.tenant_id == tenant_id).group_by(ModelDeployment.mode).all()
                mode_counts = {r[0]: r[1] for r in mode_results}
                
                # 副本和请求统计
                stats = db.query(
                    sql_func.sum(ModelDeployment.replicas),
                    sql_func.sum(ModelDeployment.request_count),
                    sql_func.sum(ModelDeployment.error_count)
                ).filter(ModelDeployment.tenant_id == tenant_id).first()
                
                total_replicas = stats[0] or 0
                total_requests = stats[1] or 0
                total_errors = stats[2] or 0
                
                return {
                    'total_deployments': total,
                    'running_deployments': running,
                    'by_status': status_counts,
                    'by_mode': mode_counts,
                    'total_replicas': total_replicas,
                    'total_requests': total_requests,
                    'total_errors': total_errors,
                    'error_rate': round(total_errors / total_requests * 100, 2) if total_requests > 0 else 0
                }
                
        except Exception as e:
            logger.error(f"Failed to get deployment statistics: {e}")
            return {}


class ModelDeploymentLogRepository:
    """模型部署日志数据访问层"""
    
    def __init__(self, use_memory_storage: bool = False):
        """初始化日志仓库"""
        self._use_memory_storage = use_memory_storage
        
        if use_memory_storage:
            self._logs: Dict[str, Dict] = {}
            self._db_manager = None
        else:
            try:
                from backend.modules.database.manager import get_database_manager
                self._db_manager = get_database_manager()
            except ImportError:
                logger.warning("Cannot import database service, falling back to memory storage")
                self._use_memory_storage = True
                self._logs: Dict[str, Dict] = {}
                self._db_manager = None
    
    def create(self, log_data: Dict[str, Any]) -> ModelDeploymentLog:
        """创建日志"""
        try:
            log_id = log_data.get('id') or _generate_id()
            
            if self._use_memory_storage:
                log_data['id'] = log_id
                log_data['timestamp'] = datetime.utcnow().isoformat()
                log_data['created_at'] = datetime.utcnow().isoformat()
                self._logs[log_id] = log_data
                return log_data
            
            with self._db_manager.get_db_session() as db:
                details = log_data.get('details', {})
                if isinstance(details, dict):
                    details = json.dumps(details)
                
                log = ModelDeploymentLog(
                    id=log_id,
                    deployment_id=log_data['deployment_id'],
                    level=log_data.get('level', 'info'),
                    action=log_data['action'],
                    message=log_data['message'],
                    details=details,
                    source=log_data.get('source')
                )
                db.add(log)
                db.commit()
                db.refresh(log)
                return log
                
        except Exception as e:
            logger.error(f"Failed to create deployment log: {e}")
            raise DatabaseError(f"Failed to create deployment log: {e}", operation="create_deployment_log")
    
    def get_by_deployment(self, deployment_id: str, level: Optional[str] = None,
                         action: Optional[str] = None, limit: int = 1000,
                         offset: int = 0) -> Tuple[List[ModelDeploymentLog], int]:
        """获取部署的日志"""
        try:
            if self._use_memory_storage:
                logs = [l for l in self._logs.values() if str(l.get('deployment_id')) == str(deployment_id)]
                if level:
                    logs = [l for l in logs if l.get('level') == level]
                if action:
                    logs = [l for l in logs if l.get('action') == action]
                logs.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
                total = len(logs)
                return logs[offset:offset + limit], total
            
            with self._db_manager.get_db_session() as db:
                query = db.query(ModelDeploymentLog).filter(ModelDeploymentLog.deployment_id == deployment_id)
                if level:
                    query = query.filter(ModelDeploymentLog.level == level)
                if action:
                    query = query.filter(ModelDeploymentLog.action == action)
                
                total = query.count()
                logs = query.order_by(ModelDeploymentLog.timestamp.desc()).offset(offset).limit(limit).all()
                return logs, total
                
        except Exception as e:
            logger.error(f"Failed to get logs by deployment: {e}")
            return [], 0
    
    def delete_by_deployment(self, deployment_id: str) -> int:
        """删除部署的所有日志"""
        try:
            if self._use_memory_storage:
                to_delete = [k for k, v in self._logs.items() if str(v.get('deployment_id')) == str(deployment_id)]
                for key in to_delete:
                    del self._logs[key]
                return len(to_delete)
            
            with self._db_manager.get_db_session() as db:
                count = db.query(ModelDeploymentLog).filter(
                    ModelDeploymentLog.deployment_id == deployment_id
                ).delete()
                db.commit()
                return count
                
        except Exception as e:
            logger.error(f"Failed to delete logs: {e}")
            return 0


class ModelServiceRepository:
    """模型服务数据访问层"""
    
    def __init__(self, use_memory_storage: bool = False):
        """初始化服务仓库"""
        self._use_memory_storage = use_memory_storage
        
        if use_memory_storage:
            self._services: Dict[str, Dict] = {}
            self._db_manager = None
        else:
            try:
                from backend.modules.database.manager import get_database_manager
                self._db_manager = get_database_manager()
            except ImportError:
                logger.warning("Cannot import database service, falling back to memory storage")
                self._use_memory_storage = True
                self._services: Dict[str, Dict] = {}
                self._db_manager = None
    
    def create(self, service_data: Dict[str, Any]) -> ModelService:
        """创建服务记录"""
        try:
            record_id = service_data.get('id') or _generate_id()
            service_id = service_data.get('service_id') or f"svc_{_generate_id()[:8]}"
            
            if self._use_memory_storage:
                service_data['id'] = record_id
                service_data['service_id'] = service_id
                service_data['created_at'] = datetime.utcnow().isoformat()
                service_data['updated_at'] = datetime.utcnow().isoformat()
                service_data.setdefault('status', 'pending')
                service_data.setdefault('request_count', 0)
                service_data.setdefault('success_count', 0)
                service_data.setdefault('error_count', 0)
                self._services[record_id] = service_data
                return service_data
            
            with self._db_manager.get_db_session() as db:
                endpoints = service_data.get('endpoints', [])
                if isinstance(endpoints, list):
                    endpoints = json.dumps(endpoints)
                
                tags = service_data.get('tags', [])
                if isinstance(tags, list):
                    tags = json.dumps(tags)
                
                extra_data = service_data.get('extra_data', {})
                if isinstance(extra_data, dict):
                    extra_data = json.dumps(extra_data)
                
                service = ModelService(
                    id=record_id,
                    tenant_id=service_data.get('tenant_id'),
                    user_id=service_data['user_id'],
                    service_id=service_id,
                    model_id=service_data['model_id'],
                    deployment_id=service_data.get('deployment_id'),
                    api_type=service_data.get('api_type', 'rest'),
                    service_mesh=service_data.get('service_mesh', False),
                    load_balancing=service_data.get('load_balancing', True),
                    circuit_breaker=service_data.get('circuit_breaker', True),
                    rate_limiting=service_data.get('rate_limiting', True),
                    timeout=service_data.get('timeout', 30),
                    max_concurrent_requests=service_data.get('max_concurrent_requests', 100),
                    endpoints=endpoints,
                    status=service_data.get('status', 'pending'),
                    tags=tags,
                    extra_data=extra_data
                )
                db.add(service)
                db.commit()
                db.refresh(service)
                return service
                
        except Exception as e:
            logger.error(f"Failed to create service: {e}")
            raise DatabaseError(f"Failed to create service: {e}", operation="create_service")
    
    def get_by_id(self, record_id: str) -> Optional[ModelService]:
        """根据ID获取服务记录"""
        try:
            if self._use_memory_storage:
                return self._services.get(record_id)
            
            with self._db_manager.get_db_session() as db:
                return db.query(ModelService).filter(ModelService.id == record_id).first()
                
        except Exception as e:
            logger.error(f"Failed to get service by id: {e}")
            return None
    
    def get_by_service_id(self, service_id: str) -> Optional[ModelService]:
        """根据服务ID获取服务记录"""
        try:
            if self._use_memory_storage:
                for svc in self._services.values():
                    if svc.get('service_id') == service_id:
                        return svc
                return None
            
            with self._db_manager.get_db_session() as db:
                return db.query(ModelService).filter(
                    ModelService.service_id == service_id
                ).first()
                
        except Exception as e:
            logger.error(f"Failed to get service by service_id: {e}")
            return None
    
    def get_by_tenant(self, tenant_id: str, model_id: Optional[str] = None,
                     status: Optional[str] = None, api_type: Optional[str] = None,
                     limit: int = 100, offset: int = 0) -> Tuple[List[ModelService], int]:
        """获取租户的服务列表"""
        try:
            if self._use_memory_storage:
                services = [s for s in self._services.values() if s.get('tenant_id') == tenant_id]
                if model_id:
                    services = [s for s in services if s.get('model_id') == model_id]
                if status:
                    services = [s for s in services if s.get('status') == status]
                if api_type:
                    services = [s for s in services if s.get('api_type') == api_type]
                
                services.sort(key=lambda x: x.get('created_at', ''), reverse=True)
                total = len(services)
                return services[offset:offset + limit], total
            
            with self._db_manager.get_db_session() as db:
                query = db.query(ModelService).filter(ModelService.tenant_id == tenant_id)
                
                if model_id:
                    query = query.filter(ModelService.model_id == model_id)
                if status:
                    query = query.filter(ModelService.status == status)
                if api_type:
                    query = query.filter(ModelService.api_type == api_type)
                
                total = query.count()
                services = query.order_by(ModelService.created_at.desc()).offset(offset).limit(limit).all()
                return services, total
                
        except Exception as e:
            logger.error(f"Failed to get services by tenant: {e}")
            return [], 0
    
    def update(self, record_id: str, update_data: Dict[str, Any]) -> Optional[ModelService]:
        """更新服务记录"""
        try:
            if self._use_memory_storage:
                if record_id in self._services:
                    self._services[record_id].update(update_data)
                    self._services[record_id]['updated_at'] = datetime.utcnow().isoformat()
                    return self._services[record_id]
                return None
            
            with self._db_manager.get_db_session() as db:
                service = db.query(ModelService).filter(ModelService.id == record_id).first()
                if not service:
                    return None
                
                for key, value in update_data.items():
                    if key in ('endpoints', 'tags', 'extra_data') and isinstance(value, (dict, list)):
                        value = json.dumps(value)
                    if hasattr(service, key):
                        setattr(service, key, value)
                
                service.updated_at = datetime.utcnow()
                db.commit()
                db.refresh(service)
                return service
                
        except Exception as e:
            logger.error(f"Failed to update service: {e}")
            raise DatabaseError(f"Failed to update service: {e}", operation="update_service")
    
    def update_by_service_id(self, service_id: str, update_data: Dict[str, Any]) -> Optional[ModelService]:
        """根据服务ID更新记录"""
        try:
            if self._use_memory_storage:
                for record_id, svc in self._services.items():
                    if svc.get('service_id') == service_id:
                        return self.update(record_id, update_data)
                return None
            
            with self._db_manager.get_db_session() as db:
                service = db.query(ModelService).filter(
                    ModelService.service_id == service_id
                ).first()
                if not service:
                    return None
                
                for key, value in update_data.items():
                    if key in ('endpoints', 'tags', 'extra_data') and isinstance(value, (dict, list)):
                        value = json.dumps(value)
                    if hasattr(service, key):
                        setattr(service, key, value)
                
                service.updated_at = datetime.utcnow()
                db.commit()
                db.refresh(service)
                return service
                
        except Exception as e:
            logger.error(f"Failed to update service: {e}")
            raise DatabaseError(f"Failed to update service: {e}", operation="update_service")
    
    def delete(self, record_id: str) -> bool:
        """删除服务记录"""
        try:
            if self._use_memory_storage:
                if record_id in self._services:
                    del self._services[record_id]
                    return True
                return False
            
            with self._db_manager.get_db_session() as db:
                service = db.query(ModelService).filter(ModelService.id == record_id).first()
                if not service:
                    return False
                
                db.delete(service)
                db.commit()
                return True
                
        except Exception as e:
            logger.error(f"Failed to delete service: {e}")
            return False
    
    def get_statistics(self, tenant_id: str) -> Dict[str, Any]:
        """获取服务统计信息"""
        try:
            if self._use_memory_storage:
                services = [s for s in self._services.values() if s.get('tenant_id') == tenant_id]
                
                status_counts = {}
                api_type_counts = {}
                total_requests = 0
                total_success = 0
                total_errors = 0
                
                for s in services:
                    status = s.get('status', 'unknown')
                    status_counts[status] = status_counts.get(status, 0) + 1
                    
                    api_type = s.get('api_type', 'unknown')
                    api_type_counts[api_type] = api_type_counts.get(api_type, 0) + 1
                    
                    total_requests += s.get('request_count', 0)
                    total_success += s.get('success_count', 0)
                    total_errors += s.get('error_count', 0)
                
                return {
                    'total_services': len(services),
                    'active_services': status_counts.get('active', 0),
                    'by_status': status_counts,
                    'by_api_type': api_type_counts,
                    'total_requests': total_requests,
                    'total_success': total_success,
                    'total_errors': total_errors,
                    'success_rate': round(total_success / total_requests * 100, 2) if total_requests > 0 else 0
                }
            
            with self._db_manager.get_db_session() as db:
                from sqlalchemy import func as sql_func
                
                total = db.query(sql_func.count(ModelService.id)).filter(
                    ModelService.tenant_id == tenant_id
                ).scalar() or 0
                
                active = db.query(sql_func.count(ModelService.id)).filter(
                    ModelService.tenant_id == tenant_id,
                    ModelService.status == 'active'
                ).scalar() or 0
                
                status_results = db.query(
                    ModelService.status,
                    sql_func.count(ModelService.id)
                ).filter(ModelService.tenant_id == tenant_id).group_by(ModelService.status).all()
                status_counts = {r[0]: r[1] for r in status_results}
                
                api_type_results = db.query(
                    ModelService.api_type,
                    sql_func.count(ModelService.id)
                ).filter(ModelService.tenant_id == tenant_id).group_by(ModelService.api_type).all()
                api_type_counts = {r[0]: r[1] for r in api_type_results}
                
                stats = db.query(
                    sql_func.sum(ModelService.request_count),
                    sql_func.sum(ModelService.success_count),
                    sql_func.sum(ModelService.error_count)
                ).filter(ModelService.tenant_id == tenant_id).first()
                
                total_requests = stats[0] or 0
                total_success = stats[1] or 0
                total_errors = stats[2] or 0
                
                return {
                    'total_services': total,
                    'active_services': active,
                    'by_status': status_counts,
                    'by_api_type': api_type_counts,
                    'total_requests': total_requests,
                    'total_success': total_success,
                    'total_errors': total_errors,
                    'success_rate': round(total_success / total_requests * 100, 2) if total_requests > 0 else 0
                }
                
        except Exception as e:
            logger.error(f"Failed to get service statistics: {e}")
            return {}


class DeploymentModeConfigRepository:
    """部署模式配置仓库
    
    管理部署模式和发布策略的配置信息
    """
    
    def __init__(self, use_memory_storage: bool = False):
        """初始化仓库"""
        self._use_memory_storage = use_memory_storage
        self._memory_storage: Dict[str, Dict[str, Any]] = {}
        self._initialized = False
        
        # 系统内置的部署模式定义
        self._builtin_modes = [
            {
                'config_type': 'mode',
                'code': 'online',
                'name': '在线服务部署',
                'description': '将模型部署为实时在线服务，支持低延迟推理请求。适用于需要实时响应的业务场景，如API服务、Web应用等。',
                'icon': 'cloud-server',
                'category': 'production',
                'default_config': {
                    'replicas': 2,
                    'cpu_limit': '2000m',
                    'memory_limit': '4Gi',
                    'health_check': True,
                    'autoscaling': False
                },
                'required_resources': {
                    'min_cpu': '500m',
                    'min_memory': '1Gi',
                    'supports_gpu': True
                },
                'supported_features': ['autoscaling', 'health_check', 'monitoring', 'load_balancing', 'circuit_breaker'],
                'limitations': ['需要稳定的网络环境', '实时性要求高'],
                'min_replicas': 1,
                'max_replicas': 100,
                'requires_gpu': False,
                'recommended_scenarios': ['实时推理', 'API服务', 'Web应用', '移动应用后端'],
                'is_default': True,
                'sort_order': 1
            },
            {
                'config_type': 'mode',
                'code': 'batch',
                'name': '批处理部署',
                'description': '支持大规模数据的批量处理推理，优化吞吐量。适用于数据分析、报表生成、数据清洗等离线处理场景。',
                'icon': 'database',
                'category': 'batch',
                'default_config': {
                    'replicas': 1,
                    'cpu_limit': '4000m',
                    'memory_limit': '8Gi',
                    'batch_size': 32,
                    'timeout_seconds': 3600
                },
                'required_resources': {
                    'min_cpu': '1000m',
                    'min_memory': '2Gi',
                    'supports_gpu': True
                },
                'supported_features': ['batch_processing', 'scheduling', 'retry', 'monitoring'],
                'limitations': ['非实时处理', '延迟较高'],
                'min_replicas': 1,
                'max_replicas': 50,
                'requires_gpu': False,
                'recommended_scenarios': ['数据分析', '报表生成', 'ETL处理', '大规模推理'],
                'is_default': False,
                'sort_order': 2
            },
            {
                'config_type': 'mode',
                'code': 'edge',
                'name': '边缘部署',
                'description': '部署到边缘设备或近端服务器，减少网络延迟。适用于IoT设备、本地化处理、隐私敏感场景。',
                'icon': 'router',
                'category': 'edge',
                'default_config': {
                    'replicas': 1,
                    'cpu_limit': '1000m',
                    'memory_limit': '2Gi',
                    'model_optimization': True,
                    'quantization': 'int8'
                },
                'required_resources': {
                    'min_cpu': '250m',
                    'min_memory': '512Mi',
                    'supports_gpu': False
                },
                'supported_features': ['model_compression', 'quantization', 'offline_mode', 'local_cache'],
                'limitations': ['资源受限', '需要模型优化', '更新困难'],
                'min_replicas': 1,
                'max_replicas': 10,
                'requires_gpu': False,
                'recommended_scenarios': ['IoT设备', '移动端推理', '隐私保护', '低延迟需求'],
                'is_default': False,
                'sort_order': 3
            },
            {
                'config_type': 'mode',
                'code': 'hybrid',
                'name': '混合部署',
                'description': '结合云端和边缘部署的优势，支持灵活的资源调度。适用于需要弹性扩展和本地处理能力的复杂场景。',
                'icon': 'cluster',
                'category': 'hybrid',
                'default_config': {
                    'replicas': 2,
                    'cpu_limit': '2000m',
                    'memory_limit': '4Gi',
                    'edge_replicas': 1,
                    'cloud_ratio': 0.7
                },
                'required_resources': {
                    'min_cpu': '500m',
                    'min_memory': '1Gi',
                    'supports_gpu': True
                },
                'supported_features': ['autoscaling', 'edge_sync', 'load_balancing', 'failover', 'monitoring'],
                'limitations': ['架构复杂', '需要协调多个环境'],
                'min_replicas': 1,
                'max_replicas': 50,
                'requires_gpu': False,
                'recommended_scenarios': ['弹性业务', '多环境部署', '高可用需求', '混合云架构'],
                'is_default': False,
                'sort_order': 4
            }
        ]
        
        # 系统内置的发布策略定义
        self._builtin_strategies = [
            {
                'config_type': 'strategy',
                'code': 'rolling',
                'name': '滚动更新',
                'description': '逐步替换旧版本实例，确保服务不中断。适用于常规更新场景，平衡更新速度和服务稳定性。',
                'icon': 'sync',
                'category': 'standard',
                'default_config': {
                    'max_unavailable': 1,
                    'max_surge': 1,
                    'step_percent': 25,
                    'wait_seconds': 30
                },
                'required_resources': {},
                'supported_features': ['zero_downtime', 'gradual_rollout', 'automatic_rollback'],
                'limitations': ['更新时间较长', '需要足够的资源缓冲'],
                'min_replicas': 2,
                'max_replicas': 100,
                'requires_gpu': False,
                'recommended_scenarios': ['常规更新', '生产环境', '风险敏感场景'],
                'is_default': True,
                'sort_order': 1
            },
            {
                'config_type': 'strategy',
                'code': 'canary',
                'name': '金丝雀发布',
                'description': '先将新版本部署到小部分流量，验证无误后逐步扩大。适用于风险较高的更新，需要逐步验证的场景。',
                'icon': 'experiment',
                'category': 'progressive',
                'default_config': {
                    'initial_percent': 5,
                    'increment_percent': 10,
                    'analysis_interval_seconds': 300,
                    'success_threshold': 0.99
                },
                'required_resources': {},
                'supported_features': ['traffic_split', 'metrics_analysis', 'automatic_promotion', 'rollback'],
                'limitations': ['需要流量管理能力', '验证周期较长'],
                'min_replicas': 2,
                'max_replicas': 100,
                'requires_gpu': False,
                'recommended_scenarios': ['重要功能发布', '风险控制', '渐进式发布'],
                'is_default': False,
                'sort_order': 2
            },
            {
                'config_type': 'strategy',
                'code': 'blue_green',
                'name': '蓝绿部署',
                'description': '维护两套完全相同的环境，一键切换流量。适用于需要快速回滚和零停机切换的关键业务。',
                'icon': 'swap',
                'category': 'instant',
                'default_config': {
                    'switch_policy': 'instant',
                    'warmup_seconds': 60,
                    'keep_old_version_hours': 24
                },
                'required_resources': {
                    'resource_multiplier': 2
                },
                'supported_features': ['instant_switch', 'instant_rollback', 'zero_downtime', 'environment_isolation'],
                'limitations': ['资源消耗翻倍', '需要两套完整环境'],
                'min_replicas': 1,
                'max_replicas': 50,
                'requires_gpu': False,
                'recommended_scenarios': ['关键业务', '快速回滚需求', '零停机更新'],
                'is_default': False,
                'sort_order': 3
            },
            {
                'config_type': 'strategy',
                'code': 'ab_testing',
                'name': 'A/B测试',
                'description': '同时运行多个版本，按比例分配流量，收集对比数据。适用于需要比较不同版本效果的实验场景。',
                'icon': 'split',
                'category': 'experimental',
                'default_config': {
                    'default_split': 50,
                    'experiment_duration_hours': 168,
                    'min_sample_size': 1000,
                    'metrics': ['latency', 'error_rate', 'conversion']
                },
                'required_resources': {},
                'supported_features': ['traffic_split', 'metrics_collection', 'statistical_analysis', 'winner_selection'],
                'limitations': ['需要足够的流量', '实验周期较长'],
                'min_replicas': 2,
                'max_replicas': 100,
                'requires_gpu': False,
                'recommended_scenarios': ['模型对比', '功能实验', '效果验证'],
                'is_default': False,
                'sort_order': 4
            }
        ]
        
        # 初始化内存存储
        if self._use_memory_storage:
            self._init_memory_storage()
    
    def _init_memory_storage(self):
        """初始化内存存储"""
        if self._initialized:
            return
        
        # 添加内置配置
        for config in self._builtin_modes + self._builtin_strategies:
            config_id = f"builtin_{config['config_type']}_{config['code']}"
            self._memory_storage[config_id] = {
                'id': config_id,
                'tenant_id': None,  # 系统级配置
                'is_system': True,
                'is_enabled': True,
                **config
            }
        
        self._initialized = True
    
    def get_all_modes(self, tenant_id: Optional[str] = None, include_disabled: bool = False) -> List[Dict[str, Any]]:
        """获取所有部署模式
        
        Args:
            tenant_id: 租户ID，用于获取租户自定义配置
            include_disabled: 是否包含已禁用的配置
            
        Returns:
            部署模式配置列表
        """
        if self._use_memory_storage:
            modes = [
                cfg for cfg in self._memory_storage.values()
                if cfg['config_type'] == 'mode' and (include_disabled or cfg.get('is_enabled', True))
            ]
            # 按排序顺序排序
            modes.sort(key=lambda x: x.get('sort_order', 999))
            return modes
        
        try:
            from backend.core.database import get_db_session
            from backend.schemas.training_models import DeploymentModeConfig
            
            with get_db_session() as db:
                query = db.query(DeploymentModeConfig).filter(
                    DeploymentModeConfig.config_type == 'mode'
                )
                
                if not include_disabled:
                    query = query.filter(DeploymentModeConfig.is_enabled == True)
                
                # 系统配置或租户特定配置
                if tenant_id:
                    query = query.filter(
                        (DeploymentModeConfig.tenant_id == None) | 
                        (DeploymentModeConfig.tenant_id == tenant_id)
                    )
                else:
                    query = query.filter(DeploymentModeConfig.tenant_id == None)
                
                configs = query.order_by(DeploymentModeConfig.sort_order).all()
                
                # 如果数据库为空，返回内置配置
                if not configs:
                    return [
                        {**cfg, 'is_system': True, 'is_enabled': True}
                        for cfg in self._builtin_modes
                    ]
                
                return [cfg.to_dict() for cfg in configs]
                
        except Exception as e:
            logger.error(f"Failed to get deployment modes: {e}")
            # 降级返回内置配置
            return [
                {**cfg, 'is_system': True, 'is_enabled': True}
                for cfg in self._builtin_modes
            ]
    
    def get_all_strategies(self, tenant_id: Optional[str] = None, include_disabled: bool = False) -> List[Dict[str, Any]]:
        """获取所有发布策略
        
        Args:
            tenant_id: 租户ID
            include_disabled: 是否包含已禁用的配置
            
        Returns:
            发布策略配置列表
        """
        if self._use_memory_storage:
            strategies = [
                cfg for cfg in self._memory_storage.values()
                if cfg['config_type'] == 'strategy' and (include_disabled or cfg.get('is_enabled', True))
            ]
            strategies.sort(key=lambda x: x.get('sort_order', 999))
            return strategies
        
        try:
            from backend.core.database import get_db_session
            from backend.schemas.training_models import DeploymentModeConfig
            
            with get_db_session() as db:
                query = db.query(DeploymentModeConfig).filter(
                    DeploymentModeConfig.config_type == 'strategy'
                )
                
                if not include_disabled:
                    query = query.filter(DeploymentModeConfig.is_enabled == True)
                
                if tenant_id:
                    query = query.filter(
                        (DeploymentModeConfig.tenant_id == None) | 
                        (DeploymentModeConfig.tenant_id == tenant_id)
                    )
                else:
                    query = query.filter(DeploymentModeConfig.tenant_id == None)
                
                configs = query.order_by(DeploymentModeConfig.sort_order).all()
                
                if not configs:
                    return [
                        {**cfg, 'is_system': True, 'is_enabled': True}
                        for cfg in self._builtin_strategies
                    ]
                
                return [cfg.to_dict() for cfg in configs]
                
        except Exception as e:
            logger.error(f"Failed to get release strategies: {e}")
            return [
                {**cfg, 'is_system': True, 'is_enabled': True}
                for cfg in self._builtin_strategies
            ]
    
    def get_mode_by_code(self, code: str, tenant_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """根据代码获取部署模式
        
        Args:
            code: 部署模式代码
            tenant_id: 租户ID
            
        Returns:
            部署模式配置或None
        """
        if self._use_memory_storage:
            for cfg in self._memory_storage.values():
                if cfg['config_type'] == 'mode' and cfg['code'] == code:
                    return cfg
            return None
        
        try:
            from backend.core.database import get_db_session
            from backend.schemas.training_models import DeploymentModeConfig
            
            with get_db_session() as db:
                query = db.query(DeploymentModeConfig).filter(
                    DeploymentModeConfig.config_type == 'mode',
                    DeploymentModeConfig.code == code,
                    DeploymentModeConfig.is_enabled == True
                )
                
                if tenant_id:
                    # 优先获取租户配置
                    config = query.filter(DeploymentModeConfig.tenant_id == tenant_id).first()
                    if not config:
                        config = query.filter(DeploymentModeConfig.tenant_id == None).first()
                else:
                    config = query.filter(DeploymentModeConfig.tenant_id == None).first()
                
                if config:
                    return config.to_dict()
                
                # 回退到内置配置
                for cfg in self._builtin_modes:
                    if cfg['code'] == code:
                        return {**cfg, 'is_system': True, 'is_enabled': True}
                
                return None
                
        except Exception as e:
            logger.error(f"Failed to get mode by code {code}: {e}")
            for cfg in self._builtin_modes:
                if cfg['code'] == code:
                    return {**cfg, 'is_system': True, 'is_enabled': True}
            return None
    
    def get_strategy_by_code(self, code: str, tenant_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """根据代码获取发布策略
        
        Args:
            code: 发布策略代码
            tenant_id: 租户ID
            
        Returns:
            发布策略配置或None
        """
        if self._use_memory_storage:
            for cfg in self._memory_storage.values():
                if cfg['config_type'] == 'strategy' and cfg['code'] == code:
                    return cfg
            return None
        
        try:
            from backend.core.database import get_db_session
            from backend.schemas.training_models import DeploymentModeConfig
            
            with get_db_session() as db:
                query = db.query(DeploymentModeConfig).filter(
                    DeploymentModeConfig.config_type == 'strategy',
                    DeploymentModeConfig.code == code,
                    DeploymentModeConfig.is_enabled == True
                )
                
                if tenant_id:
                    config = query.filter(DeploymentModeConfig.tenant_id == tenant_id).first()
                    if not config:
                        config = query.filter(DeploymentModeConfig.tenant_id == None).first()
                else:
                    config = query.filter(DeploymentModeConfig.tenant_id == None).first()
                
                if config:
                    return config.to_dict()
                
                for cfg in self._builtin_strategies:
                    if cfg['code'] == code:
                        return {**cfg, 'is_system': True, 'is_enabled': True}
                
                return None
                
        except Exception as e:
            logger.error(f"Failed to get strategy by code {code}: {e}")
            for cfg in self._builtin_strategies:
                if cfg['code'] == code:
                    return {**cfg, 'is_system': True, 'is_enabled': True}
            return None
    
    def get_mode_usage_statistics(self, tenant_id: Optional[str] = None) -> Dict[str, Any]:
        """获取部署模式使用统计
        
        Args:
            tenant_id: 租户ID
            
        Returns:
            使用统计数据
        """
        if self._use_memory_storage:
            return {
                'total_modes': len(self._builtin_modes),
                'total_strategies': len(self._builtin_strategies),
                'most_used_mode': 'online',
                'most_used_strategy': 'rolling',
                'usage_by_mode': {m['code']: 0 for m in self._builtin_modes},
                'usage_by_strategy': {s['code']: 0 for s in self._builtin_strategies}
            }
        
        try:
            from backend.core.database import get_db_session
            from backend.schemas.training_models import ModelDeployment
            from sqlalchemy import func as sql_func
            
            with get_db_session() as db:
                query = db.query(ModelDeployment)
                if tenant_id:
                    query = query.filter(ModelDeployment.tenant_id == tenant_id)
                
                # 按模式统计
                mode_stats = query.group_by(ModelDeployment.mode).with_entities(
                    ModelDeployment.mode,
                    sql_func.count(ModelDeployment.id)
                ).all()
                
                # 按策略统计
                strategy_stats = query.group_by(ModelDeployment.release_strategy).with_entities(
                    ModelDeployment.release_strategy,
                    sql_func.count(ModelDeployment.id)
                ).all()
                
                usage_by_mode = {r[0]: r[1] for r in mode_stats}
                usage_by_strategy = {r[0]: r[1] for r in strategy_stats}
                
                most_used_mode = max(usage_by_mode, key=usage_by_mode.get) if usage_by_mode else 'online'
                most_used_strategy = max(usage_by_strategy, key=usage_by_strategy.get) if usage_by_strategy else 'rolling'
                
                return {
                    'total_modes': len(self._builtin_modes),
                    'total_strategies': len(self._builtin_strategies),
                    'most_used_mode': most_used_mode,
                    'most_used_strategy': most_used_strategy,
                    'usage_by_mode': usage_by_mode,
                    'usage_by_strategy': usage_by_strategy
                }
                
        except Exception as e:
            logger.error(f"Failed to get usage statistics: {e}")
            return {
                'total_modes': len(self._builtin_modes),
                'total_strategies': len(self._builtin_strategies),
                'most_used_mode': 'online',
                'most_used_strategy': 'rolling',
                'usage_by_mode': {},
                'usage_by_strategy': {}
            }
    
    def create_custom_mode(
        self,
        tenant_id: str,
        code: str,
        name: str,
        description: str,
        default_config: Dict[str, Any],
        **kwargs
    ) -> Optional[Dict[str, Any]]:
        """创建租户自定义部署模式
        
        Args:
            tenant_id: 租户ID
            code: 配置代码
            name: 显示名称
            description: 描述
            default_config: 默认配置
            **kwargs: 其他可选参数
            
        Returns:
            创建的配置或None
        """
        if self._use_memory_storage:
            config_id = f"custom_{tenant_id}_{code}"
            config = {
                'id': config_id,
                'tenant_id': tenant_id,
                'config_type': 'mode',
                'code': code,
                'name': name,
                'description': description,
                'default_config': default_config,
                'is_system': False,
                'is_enabled': True,
                **kwargs
            }
            self._memory_storage[config_id] = config
            return config
        
        try:
            from backend.core.database import get_db_session
            from backend.schemas.training_models import DeploymentModeConfig
            
            with get_db_session() as db:
                config = DeploymentModeConfig(
                    tenant_id=tenant_id,
                    config_type='mode',
                    code=code,
                    name=name,
                    description=description,
                    default_config=default_config,
                    is_system=False,
                    is_enabled=True,
                    **kwargs
                )
                db.add(config)
                db.commit()
                db.refresh(config)
                return config.to_dict()
                
        except Exception as e:
            logger.error(f"Failed to create custom mode: {e}")
            return None
    
    def update_mode_config(
        self,
        config_id: str,
        tenant_id: str,
        updates: Dict[str, Any]
    ) -> bool:
        """更新部署模式配置
        
        只允许更新非系统配置
        
        Args:
            config_id: 配置ID
            tenant_id: 租户ID
            updates: 更新内容
            
        Returns:
            是否更新成功
        """
        if self._use_memory_storage:
            if config_id in self._memory_storage:
                cfg = self._memory_storage[config_id]
                if cfg.get('is_system', True):
                    logger.warning("Cannot update system configuration")
                    return False
                if cfg.get('tenant_id') != tenant_id:
                    logger.warning("Cannot update other tenant's configuration")
                    return False
                cfg.update(updates)
                return True
            return False
        
        try:
            from backend.core.database import get_db_session
            from backend.schemas.training_models import DeploymentModeConfig
            
            with get_db_session() as db:
                config = db.query(DeploymentModeConfig).filter(
                    DeploymentModeConfig.id == config_id,
                    DeploymentModeConfig.tenant_id == tenant_id,
                    DeploymentModeConfig.is_system == False
                ).first()
                
                if not config:
                    return False
                
                for key, value in updates.items():
                    if hasattr(config, key):
                        setattr(config, key, value)
                
                db.commit()
                return True
                
        except Exception as e:
            logger.error(f"Failed to update mode config: {e}")
            return False


class DeploymentAuditEventRepository:
    """部署审计事件数据访问层
    
    管理部署操作的审计事件记录
    """
    
    def __init__(self, use_memory_storage: bool = False):
        """初始化仓库"""
        self._use_memory_storage = use_memory_storage
        self._memory_storage: Dict[str, Dict[str, Any]] = {}
        
        if not use_memory_storage:
            try:
                from backend.modules.database.manager import get_database_manager
                self._db_manager = get_database_manager()
            except ImportError:
                logger.warning("Cannot import database service, falling back to memory storage")
                self._use_memory_storage = True
                self._db_manager = None
        else:
            self._db_manager = None
    
    def create(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """创建审计事件
        
        Args:
            event_data: 事件数据
            
        Returns:
            创建的事件记录
        """
        try:
            event_id = event_data.get('id') or _generate_id()
            
            if self._use_memory_storage:
                event_data['id'] = event_id
                event_data['event_time'] = event_data.get('event_time') or datetime.utcnow().isoformat()
                event_data['created_at'] = datetime.utcnow().isoformat()
                event_data.setdefault('status', 'success')
                event_data.setdefault('source', 'api')
                self._memory_storage[event_id] = event_data
                return event_data
            
            from backend.schemas.training_models import DeploymentAuditEvent
            
            with self._db_manager.get_db_session() as db:
                # 处理 JSON 字段
                old_value = event_data.get('old_value', {})
                if isinstance(old_value, dict):
                    old_value = json.dumps(old_value)
                
                new_value = event_data.get('new_value', {})
                if isinstance(new_value, dict):
                    new_value = json.dumps(new_value)
                
                resource_changes = event_data.get('resource_changes', {})
                if isinstance(resource_changes, dict):
                    resource_changes = json.dumps(resource_changes)
                
                tags = event_data.get('tags', [])
                if isinstance(tags, list):
                    tags = json.dumps(tags)
                
                metadata = event_data.get('metadata', {}) or event_data.get('metadata_', {})
                if isinstance(metadata, dict):
                    metadata = json.dumps(metadata)
                
                event = DeploymentAuditEvent(
                    id=event_id,
                    tenant_id=event_data.get('tenant_id'),
                    deployment_id=event_data['deployment_id'],
                    model_id=event_data.get('model_id'),
                    user_id=event_data.get('user_id'),
                    event_type=event_data['event_type'],
                    action=event_data['action'],
                    status=event_data.get('status', 'success'),
                    message=event_data.get('message'),
                    description=event_data.get('description'),
                    old_value=old_value,
                    new_value=new_value,
                    source=event_data.get('source', 'api'),
                    trigger=event_data.get('trigger'),
                    ip_address=event_data.get('ip_address'),
                    user_agent=event_data.get('user_agent'),
                    duration_ms=event_data.get('duration_ms'),
                    from_version=event_data.get('from_version'),
                    to_version=event_data.get('to_version'),
                    resource_changes=resource_changes,
                    error_code=event_data.get('error_code'),
                    error_message=event_data.get('error_message'),
                    stack_trace=event_data.get('stack_trace'),
                    parent_event_id=event_data.get('parent_event_id'),
                    correlation_id=event_data.get('correlation_id'),
                    tags=tags,
                    metadata_=metadata
                )
                db.add(event)
                db.commit()
                db.refresh(event)
                return event.to_dict()
                
        except Exception as e:
            logger.error(f"Failed to create audit event: {e}")
            raise
    
    def get_by_id(self, event_id: str) -> Optional[Dict[str, Any]]:
        """根据ID获取审计事件"""
        try:
            if self._use_memory_storage:
                return self._memory_storage.get(event_id)
            
            from backend.schemas.training_models import DeploymentAuditEvent
            
            with self._db_manager.get_db_session() as db:
                event = db.query(DeploymentAuditEvent).filter(
                    DeploymentAuditEvent.id == event_id
                ).first()
                return event.to_dict() if event else None
                
        except Exception as e:
            logger.error(f"Failed to get audit event: {e}")
            return None
    
    def list_by_deployment(
        self,
        deployment_id: str,
        tenant_id: Optional[str] = None,
        event_type: Optional[str] = None,
        action: Optional[str] = None,
        status: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100,
        offset: int = 0
    ) -> Tuple[List[Dict[str, Any]], int]:
        """获取部署的审计事件列表
        
        Args:
            deployment_id: 部署ID
            tenant_id: 租户ID
            event_type: 事件类型过滤
            action: 动作过滤
            status: 状态过滤
            start_time: 开始时间
            end_time: 结束时间
            limit: 返回数量限制
            offset: 偏移量
            
        Returns:
            (事件列表, 总数)
        """
        try:
            if self._use_memory_storage:
                events = [
                    e for e in self._memory_storage.values()
                    if e.get('deployment_id') == deployment_id
                ]
                
                if tenant_id:
                    events = [e for e in events if e.get('tenant_id') == tenant_id]
                if event_type:
                    events = [e for e in events if e.get('event_type') == event_type]
                if action:
                    events = [e for e in events if e.get('action') == action]
                if status:
                    events = [e for e in events if e.get('status') == status]
                if start_time:
                    events = [e for e in events if e.get('event_time', '') >= start_time.isoformat()]
                if end_time:
                    events = [e for e in events if e.get('event_time', '') <= end_time.isoformat()]
                
                # 按事件时间倒序排列
                events.sort(key=lambda x: x.get('event_time', ''), reverse=True)
                total = len(events)
                return events[offset:offset + limit], total
            
            from backend.schemas.training_models import DeploymentAuditEvent
            
            with self._db_manager.get_db_session() as db:
                query = db.query(DeploymentAuditEvent).filter(
                    DeploymentAuditEvent.deployment_id == deployment_id
                )
                
                if tenant_id:
                    query = query.filter(DeploymentAuditEvent.tenant_id == tenant_id)
                if event_type:
                    query = query.filter(DeploymentAuditEvent.event_type == event_type)
                if action:
                    query = query.filter(DeploymentAuditEvent.action == action)
                if status:
                    query = query.filter(DeploymentAuditEvent.status == status)
                if start_time:
                    query = query.filter(DeploymentAuditEvent.event_time >= start_time)
                if end_time:
                    query = query.filter(DeploymentAuditEvent.event_time <= end_time)
                
                total = query.count()
                events = query.order_by(
                    DeploymentAuditEvent.event_time.desc()
                ).offset(offset).limit(limit).all()
                
                return [e.to_dict() for e in events], total
                
        except Exception as e:
            logger.error(f"Failed to list audit events: {e}")
            return [], 0
    
    def list_by_tenant(
        self,
        tenant_id: str,
        event_type: Optional[str] = None,
        action: Optional[str] = None,
        status: Optional[str] = None,
        user_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100,
        offset: int = 0
    ) -> Tuple[List[Dict[str, Any]], int]:
        """获取租户的所有审计事件
        
        Args:
            tenant_id: 租户ID
            event_type: 事件类型过滤
            action: 动作过滤
            status: 状态过滤
            user_id: 用户ID过滤
            start_time: 开始时间
            end_time: 结束时间
            limit: 返回数量限制
            offset: 偏移量
            
        Returns:
            (事件列表, 总数)
        """
        try:
            if self._use_memory_storage:
                events = [
                    e for e in self._memory_storage.values()
                    if e.get('tenant_id') == tenant_id
                ]
                
                if event_type:
                    events = [e for e in events if e.get('event_type') == event_type]
                if action:
                    events = [e for e in events if e.get('action') == action]
                if status:
                    events = [e for e in events if e.get('status') == status]
                if user_id:
                    events = [e for e in events if e.get('user_id') == user_id]
                if start_time:
                    events = [e for e in events if e.get('event_time', '') >= start_time.isoformat()]
                if end_time:
                    events = [e for e in events if e.get('event_time', '') <= end_time.isoformat()]
                
                events.sort(key=lambda x: x.get('event_time', ''), reverse=True)
                total = len(events)
                return events[offset:offset + limit], total
            
            from backend.schemas.training_models import DeploymentAuditEvent
            
            with self._db_manager.get_db_session() as db:
                query = db.query(DeploymentAuditEvent).filter(
                    DeploymentAuditEvent.tenant_id == tenant_id
                )
                
                if event_type:
                    query = query.filter(DeploymentAuditEvent.event_type == event_type)
                if action:
                    query = query.filter(DeploymentAuditEvent.action == action)
                if status:
                    query = query.filter(DeploymentAuditEvent.status == status)
                if user_id:
                    query = query.filter(DeploymentAuditEvent.user_id == user_id)
                if start_time:
                    query = query.filter(DeploymentAuditEvent.event_time >= start_time)
                if end_time:
                    query = query.filter(DeploymentAuditEvent.event_time <= end_time)
                
                total = query.count()
                events = query.order_by(
                    DeploymentAuditEvent.event_time.desc()
                ).offset(offset).limit(limit).all()
                
                return [e.to_dict() for e in events], total
                
        except Exception as e:
            logger.error(f"Failed to list tenant audit events: {e}")
            return [], 0
    
    def list_by_correlation(
        self,
        correlation_id: str,
        tenant_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """获取关联的审计事件
        
        Args:
            correlation_id: 关联ID
            tenant_id: 租户ID
            
        Returns:
            关联的事件列表
        """
        try:
            if self._use_memory_storage:
                events = [
                    e for e in self._memory_storage.values()
                    if e.get('correlation_id') == correlation_id
                ]
                if tenant_id:
                    events = [e for e in events if e.get('tenant_id') == tenant_id]
                events.sort(key=lambda x: x.get('event_time', ''))
                return events
            
            from backend.schemas.training_models import DeploymentAuditEvent
            
            with self._db_manager.get_db_session() as db:
                query = db.query(DeploymentAuditEvent).filter(
                    DeploymentAuditEvent.correlation_id == correlation_id
                )
                if tenant_id:
                    query = query.filter(DeploymentAuditEvent.tenant_id == tenant_id)
                
                events = query.order_by(DeploymentAuditEvent.event_time).all()
                return [e.to_dict() for e in events]
                
        except Exception as e:
            logger.error(f"Failed to list correlated events: {e}")
            return []
    
    def get_statistics(
        self,
        tenant_id: str,
        deployment_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """获取审计事件统计
        
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
                events = [
                    e for e in self._memory_storage.values()
                    if e.get('tenant_id') == tenant_id
                ]
                if deployment_id:
                    events = [e for e in events if e.get('deployment_id') == deployment_id]
                if start_time:
                    events = [e for e in events if e.get('event_time', '') >= start_time.isoformat()]
                if end_time:
                    events = [e for e in events if e.get('event_time', '') <= end_time.isoformat()]
                
                # 按类型统计
                by_type = {}
                by_action = {}
                by_status = {}
                by_user = {}
                
                for e in events:
                    event_type = e.get('event_type', 'unknown')
                    by_type[event_type] = by_type.get(event_type, 0) + 1
                    
                    action = e.get('action', 'unknown')
                    by_action[action] = by_action.get(action, 0) + 1
                    
                    status = e.get('status', 'unknown')
                    by_status[status] = by_status.get(status, 0) + 1
                    
                    user = e.get('user_id', 'unknown')
                    by_user[user] = by_user.get(user, 0) + 1
                
                success_count = by_status.get('success', 0)
                failure_count = by_status.get('failure', 0)
                total_count = len(events)
                
                return {
                    'total_events': total_count,
                    'success_count': success_count,
                    'failure_count': failure_count,
                    'success_rate': round(success_count / total_count * 100, 2) if total_count > 0 else 0,
                    'by_event_type': by_type,
                    'by_action': by_action,
                    'by_status': by_status,
                    'by_user': by_user,
                    'top_actions': sorted(by_action.items(), key=lambda x: x[1], reverse=True)[:10]
                }
            
            from backend.schemas.training_models import DeploymentAuditEvent
            from sqlalchemy import func as sql_func
            
            with self._db_manager.get_db_session() as db:
                query = db.query(DeploymentAuditEvent).filter(
                    DeploymentAuditEvent.tenant_id == tenant_id
                )
                if deployment_id:
                    query = query.filter(DeploymentAuditEvent.deployment_id == deployment_id)
                if start_time:
                    query = query.filter(DeploymentAuditEvent.event_time >= start_time)
                if end_time:
                    query = query.filter(DeploymentAuditEvent.event_time <= end_time)
                
                total = query.count()
                
                # 按类型统计
                type_stats = db.query(
                    DeploymentAuditEvent.event_type,
                    sql_func.count(DeploymentAuditEvent.id)
                ).filter(
                    DeploymentAuditEvent.tenant_id == tenant_id
                ).group_by(DeploymentAuditEvent.event_type).all()
                
                # 按状态统计
                status_stats = db.query(
                    DeploymentAuditEvent.status,
                    sql_func.count(DeploymentAuditEvent.id)
                ).filter(
                    DeploymentAuditEvent.tenant_id == tenant_id
                ).group_by(DeploymentAuditEvent.status).all()
                
                by_type = {r[0]: r[1] for r in type_stats}
                by_status = {r[0]: r[1] for r in status_stats}
                
                success_count = by_status.get('success', 0)
                failure_count = by_status.get('failure', 0)
                
                return {
                    'total_events': total,
                    'success_count': success_count,
                    'failure_count': failure_count,
                    'success_rate': round(success_count / total * 100, 2) if total > 0 else 0,
                    'by_event_type': by_type,
                    'by_status': by_status
                }
                
        except Exception as e:
            logger.error(f"Failed to get audit statistics: {e}")
            return {}
    
    def delete_by_deployment(self, deployment_id: str, tenant_id: Optional[str] = None) -> int:
        """删除部署的所有审计事件
        
        Args:
            deployment_id: 部署ID
            tenant_id: 租户ID
            
        Returns:
            删除的事件数量
        """
        try:
            if self._use_memory_storage:
                to_delete = [
                    k for k, v in self._memory_storage.items()
                    if v.get('deployment_id') == deployment_id
                    and (not tenant_id or v.get('tenant_id') == tenant_id)
                ]
                for key in to_delete:
                    del self._memory_storage[key]
                return len(to_delete)
            
            from backend.schemas.training_models import DeploymentAuditEvent
            
            with self._db_manager.get_db_session() as db:
                query = db.query(DeploymentAuditEvent).filter(
                    DeploymentAuditEvent.deployment_id == deployment_id
                )
                if tenant_id:
                    query = query.filter(DeploymentAuditEvent.tenant_id == tenant_id)
                
                count = query.delete()
                db.commit()
                return count
                
        except Exception as e:
            logger.error(f"Failed to delete audit events: {e}")
            return 0
    
    def cleanup_old_events(
        self,
        tenant_id: str,
        retention_days: int = 90
    ) -> int:
        """清理过期的审计事件
        
        Args:
            tenant_id: 租户ID
            retention_days: 保留天数
            
        Returns:
            删除的事件数量
        """
        try:
            from datetime import timedelta
            cutoff_time = datetime.utcnow() - timedelta(days=retention_days)
            
            if self._use_memory_storage:
                to_delete = [
                    k for k, v in self._memory_storage.items()
                    if v.get('tenant_id') == tenant_id
                    and v.get('event_time', '') < cutoff_time.isoformat()
                ]
                for key in to_delete:
                    del self._memory_storage[key]
                return len(to_delete)
            
            from backend.schemas.training_models import DeploymentAuditEvent
            
            with self._db_manager.get_db_session() as db:
                count = db.query(DeploymentAuditEvent).filter(
                    DeploymentAuditEvent.tenant_id == tenant_id,
                    DeploymentAuditEvent.event_time < cutoff_time
                ).delete()
                db.commit()
                return count
                
        except Exception as e:
            logger.error(f"Failed to cleanup old events: {e}")
            return 0


# 全局仓库实例
_deployment_repository = None
_deployment_log_repository = None
_service_repository = None
_mode_config_repository = None
_audit_event_repository = None


def get_model_deployment_repository(use_memory_storage: bool = False) -> ModelDeploymentRepository:
    """获取模型部署仓库实例"""
    global _deployment_repository
    if _deployment_repository is None:
        _deployment_repository = ModelDeploymentRepository(use_memory_storage=use_memory_storage)
    return _deployment_repository


def get_model_deployment_log_repository(use_memory_storage: bool = False) -> ModelDeploymentLogRepository:
    """获取模型部署日志仓库实例"""
    global _deployment_log_repository
    if _deployment_log_repository is None:
        _deployment_log_repository = ModelDeploymentLogRepository(use_memory_storage=use_memory_storage)
    return _deployment_log_repository


def get_model_service_repository(use_memory_storage: bool = False) -> ModelServiceRepository:
    """获取模型服务仓库实例"""
    global _service_repository
    if _service_repository is None:
        _service_repository = ModelServiceRepository(use_memory_storage=use_memory_storage)
    return _service_repository


def get_deployment_mode_config_repository(use_memory_storage: bool = False) -> DeploymentModeConfigRepository:
    """获取部署模式配置仓库实例"""
    global _mode_config_repository
    if _mode_config_repository is None:
        _mode_config_repository = DeploymentModeConfigRepository(use_memory_storage=use_memory_storage)
    return _mode_config_repository


def get_deployment_audit_event_repository(use_memory_storage: bool = False) -> DeploymentAuditEventRepository:
    """获取部署审计事件仓库实例"""
    global _audit_event_repository
    if _audit_event_repository is None:
        _audit_event_repository = DeploymentAuditEventRepository(use_memory_storage=use_memory_storage)
    return _audit_event_repository


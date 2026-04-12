#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""模型服务

实现模型相关的业务逻辑，包括：
- 模型CRUD操作
- 模型版本管理
- 模型导出
- 模型部署
- 模型指标管理
"""

import sys
import os
import uuid
import logging
import hashlib
import threading
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from backend.core.exceptions import ValidationError
from backend.utils.validation import validate_id
from backend.services.model_service_interface import ModelServiceInterface
from backend.schemas.model import Model
from backend.repositories.model_repository import ModelRepository
from backend.modules.model.exceptions.model_exceptions import ModelNotFoundError, ModelValidationError

logger = logging.getLogger(__name__)


class ModelService(ModelServiceInterface):
    """模型服务
    
    提供完整的模型管理功能
    """
    
    def __init__(self, model_repository: ModelRepository):
        """初始化模型服务
        
        Args:
            model_repository: 模型仓库实例
        """
        self.model_repository = model_repository
        self._lock = threading.RLock()
        
        # 内存存储（用于版本、事件等）
        self._versions: Dict[str, List[Dict[str, Any]]] = {}
        self._events: Dict[str, List[Dict[str, Any]]] = {}
        self._exports: Dict[str, Dict[str, Any]] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}
        
        logger.info("ModelService initialized")
    
    # ==========================================================================
    # 模型CRUD操作
    # ==========================================================================
    
    def create_model(
        self, 
        user_id: str, 
        name: str, 
        description: Optional[str] = None,
        version: str = "1.0.0",
        model_type: str = "classification",
        architecture: str = "transformer",
        framework: str = "pytorch",
        storage_path: str = "",
        config: Optional[Dict[str, Any]] = None,
        training_session_id: Optional[str] = None,
        dataset_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        category: Optional[str] = None,
        tenant_id: Optional[str] = None
    ) -> Model:
        """创建模型
        
        Args:
            user_id: 用户ID
            name: 模型名称
            description: 模型描述
            version: 版本号
            model_type: 模型类型
            architecture: 架构类型
            framework: 框架类型
            storage_path: 存储路径
            config: 配置信息
            training_session_id: 训练会话ID
            dataset_id: 数据集ID
            tags: 标签列表
            category: 分类
            tenant_id: 租户ID
            
        Returns:
            Model: 创建的模型对象
            
        Raises:
            ModelValidationError: 当输入参数验证失败时
        """
        # 参数验证
        if not name or len(name.strip()) == 0:
            raise ModelValidationError("模型名称不能为空")
        if len(name) > 200:
            raise ModelValidationError("模型名称不能超过200个字符")
        
        try:
            # 创建模型实例
            model = Model(
                user_id=user_id,
                name=name.strip(),
                description=description,
                version=version,
                model_type=model_type,
                framework=framework,
                storage_path=storage_path,
                config=config or {},
                training_session_id=training_session_id
            )
            
            # 保存到仓库
            created_model = self.model_repository.create(model)
            
            # 记录创建事件
            self._record_event(
                model_id=created_model.id,
                event_type='created',
                event_message=f"Model '{name}' created",
                user_id=user_id
            )
            
            # 创建初始版本
            self._create_version(
                model_id=created_model.id,
                version=version,
                description="Initial version",
                created_by=user_id,
                is_active=True
            )
            
            logger.info(f"Model created: {created_model.id} ({name})")
            
            return created_model
            
        except ValidationError as e:
            raise ModelValidationError(f"创建模型失败: {str(e)}") from e
    
    def get_model(self, model_id: str) -> Optional[Model]:
        """获取模型
        
        Args:
            model_id: 模型ID
            
        Returns:
            Model: 模型对象，如果不存在则返回None
            
        Raises:
            ModelValidationError: 当ID格式不正确时
        """
        # 验证ID格式
        validate_id(model_id, "model_id")
        
        return self.model_repository.get_by_id(model_id)
    
    def get_model_detail(self, model_id: str) -> Dict[str, Any]:
        """获取模型详情（包含版本、指标等）
        
        Args:
            model_id: 模型ID
            
        Returns:
            包含完整模型信息的字典
        """
        model = self.get_model(model_id)
        if not model:
            raise ModelNotFoundError(f"模型 {model_id} 不存在")
        
        result = model.to_dict() if hasattr(model, 'to_dict') else {
            'id': model.id,
            'name': model.name,
            'description': model.description,
            'version': model.version,
            'model_type': model.model_type,
            'framework': model.framework,
            'status': model.status,
            'storage_path': model.storage_path,
            'config': model.config,
            'user_id': model.user_id,
            'created_at': model.created_at.isoformat() if model.created_at else None,
            'updated_at': model.updated_at.isoformat() if model.updated_at else None,
        }
        
        # 添加版本信息
        result['versions'] = self._get_versions(model_id)
        
        # 添加元数据
        result['metadata'] = self._metadata.get(model_id, {})
        
        # 添加最近事件
        result['recent_events'] = self._get_recent_events(model_id, limit=10)
        
        return result
    
    def list_models(
        self, 
        user_id: str, 
        limit: int = 50, 
        offset: int = 0,
        status: Optional[str] = None,
        model_type: Optional[str] = None,
        framework: Optional[str] = None,
        tags: Optional[List[str]] = None,
        search: Optional[str] = None
    ) -> List[Model]:
        """获取用户模型列表
        
        Args:
            user_id: 用户ID
            limit: 返回数量限制
            offset: 偏移量
            status: 状态过滤
            model_type: 类型过滤
            framework: 框架过滤
            tags: 标签过滤
            search: 搜索关键词
            
        Returns:
            List[Model]: 模型列表
            
        Raises:
            ModelValidationError: 当输入参数验证失败时
        """
        if limit <= 0 or limit > 100:
            raise ModelValidationError("限制数量必须在1-100之间")
            
        if offset < 0:
            raise ModelValidationError("偏移量不能为负数")
        
        models = self.model_repository.list_by_user(user_id, limit=1000, offset=0)
        
        # 应用过滤
        filtered_models = []
        for model in models:
            if status and getattr(model, 'status', None) != status:
                continue
            if model_type and getattr(model, 'model_type', None) != model_type:
                continue
            if framework and getattr(model, 'framework', None) != framework:
                continue
            if search:
                name = getattr(model, 'name', '') or ''
                desc = getattr(model, 'description', '') or ''
                if search.lower() not in name.lower() and search.lower() not in desc.lower():
                    continue
            filtered_models.append(model)
        
        # 应用分页
        return filtered_models[offset:offset + limit]
    
    def list_models_paginated(
        self,
        user_id: str,
        page: int = 1,
        page_size: int = 20,
        sort_by: str = 'created_at',
        sort_order: str = 'desc',
        **filters
    ) -> Dict[str, Any]:
        """分页获取模型列表
        
        Args:
            user_id: 用户ID
            page: 页码
            page_size: 每页数量
            sort_by: 排序字段
            sort_order: 排序方向
            **filters: 过滤条件
            
        Returns:
            分页结果
        """
        if page < 1:
            page = 1
        if page_size < 1 or page_size > 100:
            page_size = 20
        
        offset = (page - 1) * page_size
        
        # 获取全部模型并过滤
        all_models = self.model_repository.list_by_user(user_id, limit=10000, offset=0)
        
        # 应用过滤
        filtered = all_models
        if filters.get('status'):
            filtered = [m for m in filtered if getattr(m, 'status', None) == filters['status']]
        if filters.get('model_type'):
            filtered = [m for m in filtered if getattr(m, 'model_type', None) == filters['model_type']]
        
        total = len(filtered)
        
        # 排序
        reverse = sort_order.lower() == 'desc'
        filtered.sort(key=lambda x: getattr(x, sort_by, None) or '', reverse=reverse)
        
        # 分页
        models = filtered[offset:offset + page_size]
        
        return {
            'models': [m.to_dict() if hasattr(m, 'to_dict') else {'id': m.id, 'name': m.name} for m in models],
            'total': total,
            'page': page,
            'page_size': page_size,
            'total_pages': (total + page_size - 1) // page_size
        }
    
    def update_model(
        self, 
        model_id: str, 
        name: Optional[str] = None,
        description: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        category: Optional[str] = None
    ) -> Model:
        """更新模型
        
        Args:
            model_id: 模型ID
            name: 模型名称
            description: 模型描述
            config: 配置信息
            tags: 标签
            category: 分类
            
        Returns:
            Model: 更新后的模型对象
            
        Raises:
            ModelNotFoundError: 当模型不存在时
            ModelValidationError: 当输入参数验证失败时
        """
        # 验证ID格式
        validate_id(model_id, "model_id")
        
        # 获取现有模型
        model = self.model_repository.get_by_id(model_id)
        if not model:
            raise ModelNotFoundError(f"模型 {model_id} 不存在")
        
        # 验证名称
        if name is not None:
            if len(name.strip()) == 0:
                raise ModelValidationError("模型名称不能为空")
            if len(name) > 200:
                raise ModelValidationError("模型名称不能超过200个字符")
            model.name = name.strip()
            
        if description is not None:
            model.description = description
            
        if config is not None:
            model.config = config
            
        # 更新时间戳
        model.updated_at = datetime.utcnow()
        
        # 保存更新
        updated_model = self.model_repository.update(model)
        
        # 记录更新事件
        self._record_event(
            model_id=model_id,
            event_type='updated',
            event_message=f"Model updated",
            event_data={'updated_fields': [k for k, v in {'name': name, 'description': description, 'config': config}.items() if v is not None]}
        )
        
        return updated_model
    
    def delete_model(self, model_id: str, user_id: Optional[str] = None) -> bool:
        """删除模型
        
        Args:
            model_id: 模型ID
            user_id: 操作用户ID
            
        Returns:
            bool: 删除成功返回True，否则返回False
            
        Raises:
            ModelValidationError: 当ID格式不正确时
        """
        # 验证ID格式
        validate_id(model_id, "model_id")
        
        # 获取模型检查是否存在
        model = self.model_repository.get_by_id(model_id)
        if not model:
            return False
        
        # 检查是否已部署
        if getattr(model, 'status', None) == 'deployed':
            raise ModelValidationError("已部署的模型不能删除，请先停止部署")
        
        # 记录删除事件
        self._record_event(
            model_id=model_id,
            event_type='deleted',
            event_message=f"Model '{model.name}' deleted",
            user_id=user_id
        )
        
        # 清理相关数据
        with self._lock:
            if model_id in self._versions:
                del self._versions[model_id]
            if model_id in self._metadata:
                del self._metadata[model_id]
        
        result = self.model_repository.delete(model_id)
        
        if result:
            logger.info(f"Model deleted: {model_id}")
        
        return result
    
    # ==========================================================================
    # 模型状态管理
    # ==========================================================================
    
    def process_model(self, model_id: str) -> Model:
        """处理模型（开始训练/处理）
        
        Args:
            model_id: 模型ID
            
        Returns:
            Model: 处理后的模型对象
            
        Raises:
            ModelNotFoundError: 当模型不存在时
        """
        # 验证ID格式
        validate_id(model_id, "model_id")
        
        # 获取模型
        model = self.model_repository.get_by_id(model_id)
        if not model:
            raise ModelNotFoundError(f"模型 {model_id} 不存在")
        
        # 更新状态
        model.status = "processing"
        model.updated_at = datetime.utcnow()
        
        # 记录事件
        self._record_event(
            model_id=model_id,
            event_type='processing',
            event_message="Model processing started"
        )
        
        # 保存更新
        return self.model_repository.update(model)
    
    def validate_model(self, model_id: str, metrics: Dict[str, float]) -> Model:
        """验证模型
        
        Args:
            model_id: 模型ID
            metrics: 性能指标
            
        Returns:
            Model: 验证后的模型对象
            
        Raises:
            ModelNotFoundError: 当模型不存在时
        """
        # 验证ID格式
        validate_id(model_id, "model_id")
        
        # 获取模型
        model = self.model_repository.get_by_id(model_id)
        if not model:
            raise ModelNotFoundError(f"模型 {model_id} 不存在")
        
        # 更新验证信息
        model.status = "validated"
        model.config = model.config or {}
        model.config["validation_metrics"] = metrics
        model.updated_at = datetime.utcnow()
        
        # 保存指标
        self._update_metadata(model_id, {
            'validation_metrics': metrics,
            'validated_at': datetime.utcnow().isoformat()
        })
        
        # 记录事件
        self._record_event(
            model_id=model_id,
            event_type='validated',
            event_message="Model validation completed",
            event_data={'metrics': metrics}
        )
        
        # 保存更新
        return self.model_repository.update(model)
    
    def mark_model_ready(self, model_id: str) -> Model:
        """标记模型为就绪状态
        
        Args:
            model_id: 模型ID
            
        Returns:
            Model: 更新后的模型对象
            
        Raises:
            ModelNotFoundError: 当模型不存在时
        """
        # 验证ID格式
        validate_id(model_id, "model_id")
        
        # 获取模型
        model = self.model_repository.get_by_id(model_id)
        if not model:
            raise ModelNotFoundError(f"模型 {model_id} 不存在")
        
        # 更新状态
        model.status = "ready"
        model.updated_at = datetime.utcnow()
        
        # 记录事件
        self._record_event(
            model_id=model_id,
            event_type='ready',
            event_message="Model marked as ready"
        )
        
        # 保存更新
        return self.model_repository.update(model)
    
    def deploy_model(
        self,
        model_id: str,
        deployment_config: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None
    ) -> Model:
        """部署模型
        
        Args:
            model_id: 模型ID
            deployment_config: 部署配置
            user_id: 操作用户ID
            
        Returns:
            Model: 部署后的模型对象
            
        Raises:
            ModelNotFoundError: 当模型不存在时
            ModelValidationError: 当模型状态不允许部署时
        """
        # 验证ID格式
        validate_id(model_id, "model_id")
        
        # 获取模型
        model = self.model_repository.get_by_id(model_id)
        if not model:
            raise ModelNotFoundError(f"模型 {model_id} 不存在")
        
        # 检查状态
        current_status = getattr(model, 'status', None)
        if current_status not in ['ready', 'validated', 'trained']:
            raise ModelValidationError(f"模型状态 '{current_status}' 不允许部署，请先验证模型")
        
        # 更新状态
        model.status = "deployed"
        model.updated_at = datetime.utcnow()
        
        # 更新部署配置
        if deployment_config:
            model.config = model.config or {}
            model.config['deployment'] = deployment_config
        
        # 生成部署端点
        endpoint = f"/api/v1/inference/{model_id}"
        model.config = model.config or {}
        model.config['deployment_endpoint'] = endpoint
        
        # 记录事件
        self._record_event(
            model_id=model_id,
            event_type='deployed',
            event_message="Model deployed successfully",
            event_data={'endpoint': endpoint, 'config': deployment_config},
            user_id=user_id
        )
        
        logger.info(f"Model deployed: {model_id}, endpoint: {endpoint}")
        
        # 保存更新
        return self.model_repository.update(model)
    
    def undeploy_model(self, model_id: str, user_id: Optional[str] = None) -> Model:
        """取消部署模型
        
        Args:
            model_id: 模型ID
            user_id: 操作用户ID
            
        Returns:
            Model: 更新后的模型对象
        """
        validate_id(model_id, "model_id")
        
        model = self.model_repository.get_by_id(model_id)
        if not model:
            raise ModelNotFoundError(f"模型 {model_id} 不存在")
        
        if getattr(model, 'status', None) != 'deployed':
            raise ModelValidationError("模型未部署")
        
        model.status = "ready"
        model.updated_at = datetime.utcnow()
        
        if model.config and 'deployment_endpoint' in model.config:
            del model.config['deployment_endpoint']
        
        self._record_event(
            model_id=model_id,
            event_type='undeployed',
            event_message="Model undeployed",
            user_id=user_id
        )
        
        return self.model_repository.update(model)
    
    def archive_model(self, model_id: str, user_id: Optional[str] = None) -> Model:
        """归档模型
        
        Args:
            model_id: 模型ID
            user_id: 操作用户ID
            
        Returns:
            Model: 更新后的模型对象
        """
        validate_id(model_id, "model_id")
        
        model = self.model_repository.get_by_id(model_id)
        if not model:
            raise ModelNotFoundError(f"模型 {model_id} 不存在")
        
        if getattr(model, 'status', None) == 'deployed':
            raise ModelValidationError("已部署的模型不能归档")
        
        model.status = "archived"
        model.updated_at = datetime.utcnow()
        
        self._record_event(
            model_id=model_id,
            event_type='archived',
            event_message="Model archived",
            user_id=user_id
        )
        
        return self.model_repository.update(model)
    
    def restore_model(self, model_id: str, user_id: Optional[str] = None) -> Model:
        """恢复归档的模型
        
        Args:
            model_id: 模型ID
            user_id: 操作用户ID
            
        Returns:
            Model: 更新后的模型对象
        """
        validate_id(model_id, "model_id")
        
        model = self.model_repository.get_by_id(model_id)
        if not model:
            raise ModelNotFoundError(f"模型 {model_id} 不存在")
        
        model.status = "ready"
        model.updated_at = datetime.utcnow()
        
        self._record_event(
            model_id=model_id,
            event_type='restored',
            event_message="Model restored from archive",
            user_id=user_id
        )
        
        return self.model_repository.update(model)
    
    # ==========================================================================
    # 版本管理
    # ==========================================================================
    
    def _create_version(
        self,
        model_id: str,
        version: str,
        description: Optional[str] = None,
        created_by: Optional[str] = None,
        is_active: bool = False,
        storage_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """创建模型版本"""
        with self._lock:
            if model_id not in self._versions:
                self._versions[model_id] = []
            
            # 如果设为活动版本，取消其他版本的活动状态
            if is_active:
                for v in self._versions[model_id]:
                    v['is_active'] = False
            
            version_data = {
                'id': str(uuid.uuid4()),
                'model_id': model_id,
                'version': version,
                'description': description,
                'is_active': is_active,
                'storage_path': storage_path,
                'created_by': created_by,
                'created_at': datetime.utcnow().isoformat()
            }
            
            self._versions[model_id].append(version_data)
            
            return version_data
    
    def create_version(
        self,
        model_id: str,
        version: str,
        description: Optional[str] = None,
        user_id: Optional[str] = None,
        storage_path: Optional[str] = None,
        changelog: Optional[str] = None
    ) -> Dict[str, Any]:
        """创建新版本
        
        Args:
            model_id: 模型ID
            version: 版本号
            description: 版本描述
            user_id: 创建用户ID
            storage_path: 存储路径
            changelog: 变更日志
            
        Returns:
            创建的版本信息
        """
        validate_id(model_id, "model_id")
        
        model = self.model_repository.get_by_id(model_id)
        if not model:
            raise ModelNotFoundError(f"模型 {model_id} 不存在")
        
        # 检查版本是否已存在
        existing_versions = self._get_versions(model_id)
        for v in existing_versions:
            if v['version'] == version:
                raise ModelValidationError(f"版本 {version} 已存在")
        
        version_data = self._create_version(
            model_id=model_id,
            version=version,
            description=description,
            created_by=user_id,
            storage_path=storage_path
        )
        
        if changelog:
            version_data['changelog'] = changelog
        
        self._record_event(
            model_id=model_id,
            event_type='version_created',
            event_message=f"Version {version} created",
            user_id=user_id
        )
        
        return version_data
    
    def _get_versions(self, model_id: str) -> List[Dict[str, Any]]:
        """获取模型版本列表"""
        with self._lock:
            return self._versions.get(model_id, [])
    
    def list_versions(self, model_id: str) -> List[Dict[str, Any]]:
        """获取模型版本列表
        
        Args:
            model_id: 模型ID
            
        Returns:
            版本列表
        """
        validate_id(model_id, "model_id")
        
        model = self.model_repository.get_by_id(model_id)
        if not model:
            raise ModelNotFoundError(f"模型 {model_id} 不存在")
        
        versions = self._get_versions(model_id)
        # 按创建时间倒序
        versions.sort(key=lambda x: x.get('created_at', ''), reverse=True)
        return versions
    
    def get_version(self, model_id: str, version: str) -> Optional[Dict[str, Any]]:
        """获取特定版本
        
        Args:
            model_id: 模型ID
            version: 版本号
            
        Returns:
            版本信息
        """
        versions = self._get_versions(model_id)
        for v in versions:
            if v['version'] == version:
                return v
        return None
    
    def activate_version(self, model_id: str, version: str, user_id: Optional[str] = None) -> Dict[str, Any]:
        """激活指定版本
        
        Args:
            model_id: 模型ID
            version: 版本号
            user_id: 操作用户ID
            
        Returns:
            更新后的版本信息
        """
        validate_id(model_id, "model_id")
        
        with self._lock:
            versions = self._versions.get(model_id, [])
            target_version = None
            
            for v in versions:
                if v['version'] == version:
                    target_version = v
                    v['is_active'] = True
                else:
                    v['is_active'] = False
            
            if not target_version:
                raise ModelNotFoundError(f"版本 {version} 不存在")
        
        # 更新主模型版本
        model = self.model_repository.get_by_id(model_id)
        if model:
            model.version = version
            model.updated_at = datetime.utcnow()
            self.model_repository.update(model)
        
        self._record_event(
            model_id=model_id,
            event_type='version_activated',
            event_message=f"Version {version} activated",
            user_id=user_id
        )
        
        return target_version
    
    # ==========================================================================
    # 指标管理
    # ==========================================================================
    
    def _update_metadata(self, model_id: str, data: Dict[str, Any]):
        """更新模型元数据"""
        with self._lock:
            if model_id not in self._metadata:
                self._metadata[model_id] = {}
            self._metadata[model_id].update(data)
    
    def update_metrics(
        self,
        model_id: str,
        metrics: Dict[str, Any],
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """更新模型指标
        
        Args:
            model_id: 模型ID
            metrics: 指标数据
            user_id: 操作用户ID
            
        Returns:
            更新后的指标
        """
        validate_id(model_id, "model_id")
        
        model = self.model_repository.get_by_id(model_id)
        if not model:
            raise ModelNotFoundError(f"模型 {model_id} 不存在")
        
        self._update_metadata(model_id, {
            'metrics': metrics,
            'metrics_updated_at': datetime.utcnow().isoformat()
        })
        
        self._record_event(
            model_id=model_id,
            event_type='metrics_updated',
            event_message="Model metrics updated",
            event_data={'metrics': metrics},
            user_id=user_id
        )
        
        return self._metadata.get(model_id, {})
    
    def get_metrics(self, model_id: str) -> Dict[str, Any]:
        """获取模型指标
        
        Args:
            model_id: 模型ID
            
        Returns:
            指标数据
        """
        validate_id(model_id, "model_id")
        
        model = self.model_repository.get_by_id(model_id)
        if not model:
            raise ModelNotFoundError(f"模型 {model_id} 不存在")
        
        metadata = self._metadata.get(model_id, {})
        return metadata.get('metrics', {})
    
    # ==========================================================================
    # 导出功能
    # ==========================================================================
    
    def export_model(
        self,
        model_id: str,
        export_format: str = 'onnx',
        export_config: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """导出模型
        
        Args:
            model_id: 模型ID
            export_format: 导出格式
            export_config: 导出配置
            user_id: 操作用户ID
            
        Returns:
            导出任务信息
        """
        validate_id(model_id, "model_id")
        
        model = self.model_repository.get_by_id(model_id)
        if not model:
            raise ModelNotFoundError(f"模型 {model_id} 不存在")
        
        supported_formats = ['onnx', 'torchscript', 'tensorflow', 'tensorrt', 'safetensors']
        if export_format.lower() not in supported_formats:
            raise ModelValidationError(f"不支持的导出格式: {export_format}")
        
        export_id = str(uuid.uuid4())
        export_task = {
            'id': export_id,
            'model_id': model_id,
            'format': export_format,
            'config': export_config or {},
            'status': 'pending',
            'created_at': datetime.utcnow().isoformat(),
            'user_id': user_id
        }
        
        with self._lock:
            self._exports[export_id] = export_task
        
        # 模拟导出完成
        export_task['status'] = 'completed'
        export_task['export_path'] = f"/exports/{model_id}/{export_format}/{export_id}"
        export_task['completed_at'] = datetime.utcnow().isoformat()
        
        self._record_event(
            model_id=model_id,
            event_type='exported',
            event_message=f"Model exported to {export_format}",
            event_data={'export_id': export_id, 'format': export_format},
            user_id=user_id
        )
        
        return export_task
    
    def get_export_status(self, export_id: str) -> Optional[Dict[str, Any]]:
        """获取导出任务状态
        
        Args:
            export_id: 导出任务ID
            
        Returns:
            导出任务信息
        """
        with self._lock:
            return self._exports.get(export_id)
    
    def list_exports(self, model_id: str) -> List[Dict[str, Any]]:
        """获取模型导出历史
        
        Args:
            model_id: 模型ID
            
        Returns:
            导出记录列表
        """
        with self._lock:
            return [e for e in self._exports.values() if e['model_id'] == model_id]
    
    # ==========================================================================
    # 事件管理
    # ==========================================================================
    
    def _record_event(
        self,
        model_id: str,
        event_type: str,
        event_message: str,
        event_data: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None
    ):
        """记录模型事件"""
        with self._lock:
            if model_id not in self._events:
                self._events[model_id] = []
            
            event = {
                'id': str(uuid.uuid4()),
                'model_id': model_id,
                'event_type': event_type,
                'event_message': event_message,
                'event_data': event_data or {},
                'user_id': user_id,
                'timestamp': datetime.utcnow().isoformat()
            }
            
            self._events[model_id].append(event)
            
            # 限制事件数量
            if len(self._events[model_id]) > 1000:
                self._events[model_id] = self._events[model_id][-500:]
    
    def _get_recent_events(self, model_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """获取最近事件"""
        with self._lock:
            events = self._events.get(model_id, [])
            events.sort(key=lambda x: x['timestamp'], reverse=True)
            return events[:limit]
    
    def get_events(
        self,
        model_id: str,
        event_type: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> Dict[str, Any]:
        """获取模型事件
        
        Args:
            model_id: 模型ID
            event_type: 事件类型过滤
            limit: 返回数量
            offset: 偏移量
            
        Returns:
            事件列表和总数
        """
        validate_id(model_id, "model_id")
        
        with self._lock:
            events = self._events.get(model_id, [])
        
        if event_type:
            events = [e for e in events if e['event_type'] == event_type]
        
        events.sort(key=lambda x: x['timestamp'], reverse=True)
        total = len(events)
        events = events[offset:offset + limit]
        
        return {
            'events': events,
            'total': total,
            'limit': limit,
            'offset': offset
        }
    
    # ==========================================================================
    # 统计和摘要
    # ==========================================================================
    
    def get_summary(self, user_id: str) -> Dict[str, Any]:
        """获取用户模型摘要
        
        Args:
            user_id: 用户ID
            
        Returns:
            模型统计摘要
        """
        models = self.model_repository.list_by_user(user_id, limit=10000, offset=0)
        
        summary = {
            'total_models': len(models),
            'by_status': {},
            'by_type': {},
            'by_framework': {},
            'deployed_count': 0,
            'recent_models': []
        }
        
        for model in models:
            status = getattr(model, 'status', 'unknown')
            model_type = getattr(model, 'model_type', 'unknown')
            framework = getattr(model, 'framework', 'unknown')
            
            summary['by_status'][status] = summary['by_status'].get(status, 0) + 1
            summary['by_type'][model_type] = summary['by_type'].get(model_type, 0) + 1
            summary['by_framework'][framework] = summary['by_framework'].get(framework, 0) + 1
            
            if status == 'deployed':
                summary['deployed_count'] += 1
        
        # 最近模型
        sorted_models = sorted(
            models,
            key=lambda x: getattr(x, 'created_at', datetime.min),
            reverse=True
        )
        summary['recent_models'] = [
            {'id': m.id, 'name': m.name, 'status': getattr(m, 'status', None)}
            for m in sorted_models[:5]
        ]
        
        return summary
    
    # ==========================================================================
    # 复制和克隆
    # ==========================================================================
    
    def clone_model(
        self,
        model_id: str,
        new_name: str,
        user_id: str,
        include_versions: bool = False
    ) -> Model:
        """克隆模型
        
        Args:
            model_id: 源模型ID
            new_name: 新模型名称
            user_id: 用户ID
            include_versions: 是否包含版本历史
            
        Returns:
            克隆的模型
        """
        validate_id(model_id, "model_id")
        
        source_model = self.model_repository.get_by_id(model_id)
        if not source_model:
            raise ModelNotFoundError(f"模型 {model_id} 不存在")
        
        # 创建新模型
        new_model = self.create_model(
            user_id=user_id,
            name=new_name,
            description=f"Cloned from {source_model.name}",
            version="1.0.0",
            model_type=getattr(source_model, 'model_type', 'classification'),
            framework=getattr(source_model, 'framework', 'pytorch'),
            config=source_model.config.copy() if source_model.config else {}
        )
        
        # 复制版本
        if include_versions:
            source_versions = self._get_versions(model_id)
            for v in source_versions:
                self._create_version(
                    model_id=new_model.id,
                    version=v['version'],
                    description=v.get('description'),
                    created_by=user_id
                )
        
        self._record_event(
            model_id=new_model.id,
            event_type='cloned',
            event_message=f"Model cloned from {source_model.name}",
            event_data={'source_model_id': model_id},
            user_id=user_id
        )
        
        return new_model
    
    # ==========================================================================
    # 比较功能
    # ==========================================================================
    
    def compare_models(self, model_ids: List[str]) -> Dict[str, Any]:
        """比较多个模型
        
        Args:
            model_ids: 模型ID列表
            
        Returns:
            比较结果
        """
        if len(model_ids) < 2:
            raise ModelValidationError("至少需要2个模型进行比较")
        if len(model_ids) > 10:
            raise ModelValidationError("最多比较10个模型")
        
        models_data = []
        for model_id in model_ids:
            model = self.model_repository.get_by_id(model_id)
            if not model:
                raise ModelNotFoundError(f"模型 {model_id} 不存在")
            
            metadata = self._metadata.get(model_id, {})
            models_data.append({
                'id': model.id,
                'name': model.name,
                'version': model.version,
                'status': getattr(model, 'status', None),
                'framework': getattr(model, 'framework', None),
                'model_type': getattr(model, 'model_type', None),
                'metrics': metadata.get('metrics', {}),
                'created_at': model.created_at.isoformat() if model.created_at else None
            })
        
        return {
            'models': models_data,
            'comparison_time': datetime.utcnow().isoformat()
        }
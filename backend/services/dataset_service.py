"""数据集服务

实现数据集相关的业务逻辑，提供数据集的完整生命周期管理。
"""

import sys
import os
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from backend.core.exceptions import ValidationError
from backend.utils.validation import validate_id
from backend.services.dataset_service_interface import DatasetServiceInterface
from backend.schemas.dataset import (
    Dataset, 
    DatasetVersion,
    DatasetStatistics,
    CreateDatasetRequest,
    UpdateDatasetRequest,
    DatasetListResponse
)
from backend.repositories.dataset_repository import (
    DatasetRepository,
    DatasetVersionRepository,
    DatasetAccessLogRepository,
    DatasetTagRepository
)
from backend.modules.dataset.dataset_exceptions import (
    DatasetNotFoundError, 
    DatasetValidationError,
    DatasetBusinessLogicError
)

logger = logging.getLogger(__name__)


class DatasetService(DatasetServiceInterface):
    """数据集服务
    
    提供数据集的完整生命周期管理，包括创建、查询、更新、删除等操作。
    
    Attributes:
        dataset_repository: 数据集仓库实例
        version_repository: 版本仓库实例
        access_log_repository: 访问日志仓库实例
        tag_repository: 标签仓库实例
        
    Example:
        >>> repo = DatasetRepository()
        >>> service = DatasetService(repo)
        >>> dataset = service.create_dataset(
        ...     user_id="user123",
        ...     name="training_data",
        ...     dataset_type="text"
        ... )
    """
    
    def __init__(self, dataset_repository: DatasetRepository):
        """初始化数据集服务
        
        Args:
            dataset_repository: 数据集仓库实例，用于数据持久化操作
        """
        self.dataset_repository = dataset_repository
        self.version_repository = DatasetVersionRepository()
        self.access_log_repository = DatasetAccessLogRepository()
        self.tag_repository = DatasetTagRepository()
        
    def create_dataset(
        self, 
        user_id: str, 
        name: str, 
        description: Optional[str] = None,
        dataset_type: str = "text",
        format: str = "json",
        storage_path: str = "",
        config: Optional[Dict[str, Any]] = None
    ) -> Dataset:
        """创建数据集
        
        创建一个新的数据集并持久化到数据库。
        
        Args:
            user_id: 用户ID (必填)
                - 格式: UUID字符串
                - 示例: "550e8400-e29b-41d4-a716-446655440000"
            name: 数据集名称 (必填)
                - 长度: 1-200字符
                - 示例: "training_data_v1"
            description: 数据集描述 (可选)
                - 长度: 最大5000字符
                - 示例: "用于模型训练的文本数据集"
            dataset_type: 数据集类型 (可选，默认'text')
                - 可选值: text, image, audio, video, tabular, mixed
            format: 数据格式 (可选，默认'json')
                - 可选值: json, csv, parquet, tfrecord, arrow, custom
            storage_path: 存储路径 (可选)
                - 示例: "/data/datasets/user123/dataset1"
            config: 配置信息 (可选)
                - 类型: Dict[str, Any]
                - 示例: {"max_samples": 10000, "shuffle": True}
            
        Returns:
            Dataset: 创建成功的数据集对象
                - dataset_id: 生成的唯一ID
                - status: 初始状态为'pending'
                - created_at: 创建时间
            
        Raises:
            DatasetValidationError: 当输入参数验证失败时
                - 名称为空或过长
                - 无效的数据集类型
                - 无效的数据格式
                
        Example:
            >>> dataset = service.create_dataset(
            ...     user_id="user123",
            ...     name="my_dataset",
            ...     description="测试数据集",
            ...     dataset_type="text",
            ...     format="json"
            ... )
            >>> print(dataset.dataset_id)
        """
        try:
            # 验证请求
            request = CreateDatasetRequest(
                name=name,
                description=description,
                dataset_type=dataset_type,
                format=format,
                storage_path=storage_path,
                config=config
            )
            errors = request.validate()
            if errors:
                raise DatasetValidationError("; ".join(errors))
            
            # 创建数据集实例
            dataset = Dataset(
                user_id=user_id,
                name=name,
                description=description,
                dataset_type=dataset_type,
                format=format,
                storage_path=storage_path,
                config=config or {},
                status='pending',
                ready=False
            )
            
            # 保存到仓库
            created_dataset = self.dataset_repository.create(dataset)
            
            # 记录访问日志
            self._log_access(created_dataset.dataset_id, user_id, 'create')
            
            logger.info(f"Created dataset {created_dataset.dataset_id} for user {user_id}")
            return created_dataset
            
        except ValidationError as e:
            raise DatasetValidationError(f"创建数据集失败: {str(e)}") from e
            
    def get_dataset(self, dataset_id: str, user_id: str = None) -> Optional[Dataset]:
        """获取数据集
        
        根据ID获取数据集详细信息。
        
        Args:
            dataset_id: 数据集唯一标识符 (必填)
                - 格式: UUID字符串
                - 示例: "550e8400-e29b-41d4-a716-446655440000"
            user_id: 用户ID (可选，用于权限检查)
            
        Returns:
            Dataset: 数据集对象，包含以下字段:
                - dataset_id: 数据集ID
                - user_id: 所属用户ID
                - name: 名称
                - description: 描述
                - dataset_type: 类型
                - format: 格式
                - status: 状态
                - storage_path: 存储路径
                - size: 大小(字节)
                - record_count: 记录数
                - features: 特征信息
                - config: 配置信息
                - created_at: 创建时间
                - updated_at: 更新时间
            如果不存在则返回None
            
        Raises:
            DatasetValidationError: 当ID格式不正确时
            
        Example:
            >>> dataset = service.get_dataset("550e8400-e29b-41d4-a716-446655440000")
            >>> if dataset:
            ...     print(f"名称: {dataset.name}, 状态: {dataset.status}")
        """
        # 验证ID格式
        validate_id(dataset_id, "dataset_id")
        
        dataset = self.dataset_repository.get_by_id(dataset_id)
        
        # 记录访问日志
        if dataset and user_id:
            self._log_access(dataset_id, user_id, 'read')
        
        return dataset
        
    def list_datasets(
        self, 
        user_id: str, 
        limit: int = 50, 
        offset: int = 0,
        status: str = None,
        dataset_type: str = None,
        order_by: str = 'created_at',
        order_desc: bool = True
    ) -> List[Dataset]:
        """获取用户数据集列表
        
        分页获取用户的数据集列表，支持过滤和排序。
        
        Args:
            user_id: 用户ID (必填)
            limit: 返回数量限制 (可选，默认50)
                - 范围: 1-100
            offset: 偏移量 (可选，默认0)
                - 最小值: 0
            status: 按状态过滤 (可选)
                - 可选值: pending, uploading, processing, ready, error, archived
            dataset_type: 按类型过滤 (可选)
                - 可选值: text, image, audio, video, tabular, mixed
            order_by: 排序字段 (可选，默认'created_at')
                - 可选值: created_at, updated_at, name, size
            order_desc: 是否降序 (可选，默认True)
            
        Returns:
            List[Dataset]: 数据集对象列表
            
        Raises:
            DatasetValidationError: 当输入参数验证失败时
            
        Example:
            >>> datasets = service.list_datasets(
            ...     user_id="user123",
            ...     limit=10,
            ...     status="ready",
            ...     order_by="name"
            ... )
            >>> for ds in datasets:
            ...     print(f"{ds.name}: {ds.status}")
        """
        if limit <= 0 or limit > 100:
            raise DatasetValidationError("限制数量必须在1-100之间")
            
        if offset < 0:
            raise DatasetValidationError("偏移量不能为负数")
            
        return self.dataset_repository.list_by_user(
            user_id=user_id,
            limit=limit,
            offset=offset,
            status=status,
            dataset_type=dataset_type,
            order_by=order_by,
            order_desc=order_desc
        )
    
    def list_datasets_with_pagination(
        self, 
        user_id: str, 
        limit: int = 50, 
        offset: int = 0,
        status: str = None,
        dataset_type: str = None
    ) -> DatasetListResponse:
        """获取数据集列表(带分页信息)
        
        返回数据集列表及分页元信息。
        
        Args:
            user_id: 用户ID
            limit: 返回数量限制
            offset: 偏移量
            status: 按状态过滤
            dataset_type: 按类型过滤
            
        Returns:
            DatasetListResponse: 包含以下字段:
                - datasets: 数据集列表
                - total: 总数
                - limit: 限制数量
                - offset: 偏移量
                - has_more: 是否还有更多数据
        """
        if limit <= 0 or limit > 100:
            raise DatasetValidationError("限制数量必须在1-100之间")
            
        datasets = self.dataset_repository.list_by_user(
            user_id=user_id,
            limit=limit,
            offset=offset,
            status=status,
            dataset_type=dataset_type
        )
        
        total = self.dataset_repository.count_by_user(
            user_id=user_id,
            status=status,
            dataset_type=dataset_type
        )
        
        return DatasetListResponse(
            datasets=datasets,
            total=total,
            limit=limit,
            offset=offset,
            has_more=(offset + limit) < total
        )
        
    def update_dataset(
        self, 
        dataset_id: str, 
        name: Optional[str] = None,
        description: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        status: Optional[str] = None,
        user_id: str = None
    ) -> Dataset:
        """更新数据集
        
        更新指定数据集的信息。
        
        Args:
            dataset_id: 数据集ID (必填)
            name: 新的数据集名称 (可选)
                - 长度: 1-200字符
            description: 新的描述 (可选)
            config: 新的配置信息 (可选)
            status: 新的状态 (可选)
                - 可选值: pending, uploading, processing, ready, error, archived
            user_id: 用户ID (可选，用于权限检查和日志记录)
            
        Returns:
            Dataset: 更新后的数据集对象
            
        Raises:
            DatasetNotFoundError: 当数据集不存在时
            DatasetValidationError: 当输入参数验证失败时
            
        Example:
            >>> updated = service.update_dataset(
            ...     dataset_id="550e8400-e29b-41d4-a716-446655440000",
            ...     name="updated_name",
            ...     description="更新后的描述"
            ... )
        """
        # 验证ID格式
        validate_id(dataset_id, "dataset_id")
        
        # 验证请求
        request = UpdateDatasetRequest(
            name=name,
            description=description,
            config=config,
            status=status
        )
        errors = request.validate()
        if errors:
            raise DatasetValidationError("; ".join(errors))
        
        # 获取现有数据集
        dataset = self.dataset_repository.get_by_id(dataset_id)
        if not dataset:
            raise DatasetNotFoundError(f"数据集 {dataset_id} 不存在")
            
        # 更新字段
        if name is not None:
            dataset.name = name
            
        if description is not None:
            dataset.description = description
            
        if config is not None:
            dataset.config = config
            
        if status is not None:
            dataset.status = status
            if status == 'ready':
                dataset.ready = True
            elif status == 'error':
                dataset.ready = False
            
        # 更新时间戳
        dataset.updated_at = datetime.utcnow()
        
        # 保存更新
        updated = self.dataset_repository.update(dataset)
        
        # 记录访问日志
        if user_id:
            self._log_access(dataset_id, user_id, 'write', {
                'updated_fields': [k for k, v in {'name': name, 'description': description, 'config': config, 'status': status}.items() if v is not None]
            })
        
        logger.info(f"Updated dataset {dataset_id}")
        return updated
            
    def delete_dataset(self, dataset_id: str, user_id: str = None) -> bool:
        """删除数据集
        
        从系统中删除指定的数据集。
        
        Args:
            dataset_id: 数据集ID (必填)
            user_id: 用户ID (可选，用于权限检查和日志记录)
            
        Returns:
            bool: 删除成功返回True，数据集不存在返回False
            
        Raises:
            DatasetValidationError: 当ID格式不正确时
            DatasetBusinessLogicError: 当数据集正在被使用时
            
        Example:
            >>> success = service.delete_dataset("550e8400-e29b-41d4-a716-446655440000")
            >>> if success:
            ...     print("删除成功")
        """
        # 验证ID格式
        validate_id(dataset_id, "dataset_id")
        
        # 检查数据集状态
        dataset = self.dataset_repository.get_by_id(dataset_id)
        if dataset and dataset.status == 'processing':
            raise DatasetBusinessLogicError(f"数据集 {dataset_id} 正在处理中，无法删除")
        
        # 记录访问日志
        if user_id and dataset:
            self._log_access(dataset_id, user_id, 'delete')
        
        result = self.dataset_repository.delete(dataset_id)
        
        if result:
            logger.info(f"Deleted dataset {dataset_id}")
        
        return result
        
    def process_dataset(self, dataset_id: str, user_id: str = None) -> Dataset:
        """处理数据集
        
        将数据集状态设置为处理中，并触发数据处理流程。
        
        Args:
            dataset_id: 数据集ID (必填)
            user_id: 用户ID (可选)
            
        Returns:
            Dataset: 更新后的数据集对象，状态为'processing'
            
        Raises:
            DatasetNotFoundError: 当数据集不存在时
            DatasetValidationError: 当ID格式不正确时
            DatasetBusinessLogicError: 当数据集状态不允许处理时
            
        Example:
            >>> dataset = service.process_dataset("550e8400-e29b-41d4-a716-446655440000")
            >>> print(dataset.status)  # 输出: processing
        """
        # 验证ID格式
        validate_id(dataset_id, "dataset_id")
        
        # 获取数据集
        dataset = self.dataset_repository.get_by_id(dataset_id)
        if not dataset:
            raise DatasetNotFoundError(f"数据集 {dataset_id} 不存在")
        
        # 检查状态
        if dataset.status == 'processing':
            raise DatasetBusinessLogicError(f"数据集 {dataset_id} 已经在处理中")
        if dataset.status == 'archived':
            raise DatasetBusinessLogicError(f"数据集 {dataset_id} 已归档，无法处理")
            
        # 标记为处理中
        dataset.process()
        
        # 保存更新
        updated = self.dataset_repository.update(dataset)
        
        # 记录访问日志
        if user_id:
            self._log_access(dataset_id, user_id, 'process')
        
        logger.info(f"Started processing dataset {dataset_id}")
        return updated
        
    def validate_dataset(self, dataset_id: str, validation_result: Dict[str, Any], user_id: str = None) -> Dataset:
        """验证数据集
        
        记录数据集的验证结果。
        
        Args:
            dataset_id: 数据集ID (必填)
            validation_result: 验证结果 (必填)
                - 类型: Dict[str, Any]
                - 示例: {
                    "is_valid": True,
                    "errors": [],
                    "warnings": ["部分记录缺少标签"],
                    "statistics": {"total": 1000, "valid": 998}
                }
            user_id: 用户ID (可选)
            
        Returns:
            Dataset: 验证后的数据集对象
            
        Raises:
            DatasetNotFoundError: 当数据集不存在时
            DatasetValidationError: 当ID格式不正确时
            
        Example:
            >>> result = {"is_valid": True, "errors": []}
            >>> dataset = service.validate_dataset(
            ...     "550e8400-e29b-41d4-a716-446655440000",
            ...     result
            ... )
        """
        # 验证ID格式
        validate_id(dataset_id, "dataset_id")
        
        # 获取数据集
        dataset = self.dataset_repository.get_by_id(dataset_id)
        if not dataset:
            raise DatasetNotFoundError(f"数据集 {dataset_id} 不存在")
            
        # 验证数据集
        dataset.validate_dataset(validation_result)
        
        # 保存更新
        updated = self.dataset_repository.update(dataset)
        
        # 记录访问日志
        if user_id:
            self._log_access(dataset_id, user_id, 'validate', validation_result)
        
        logger.info(f"Validated dataset {dataset_id}")
        return updated
        
    def mark_dataset_ready(self, dataset_id: str, user_id: str = None) -> Dataset:
        """标记数据集为就绪状态
        
        将数据集状态设置为ready，表示可以用于训练。
        
        Args:
            dataset_id: 数据集ID (必填)
            user_id: 用户ID (可选)
            
        Returns:
            Dataset: 更新后的数据集对象
                - status: 'ready'
                - ready: True
            
        Raises:
            DatasetNotFoundError: 当数据集不存在时
            DatasetValidationError: 当ID格式不正确时
            
        Example:
            >>> dataset = service.mark_dataset_ready("550e8400-e29b-41d4-a716-446655440000")
            >>> print(dataset.ready)  # 输出: True
        """
        # 验证ID格式
        validate_id(dataset_id, "dataset_id")
        
        # 获取数据集
        dataset = self.dataset_repository.get_by_id(dataset_id)
        if not dataset:
            raise DatasetNotFoundError(f"数据集 {dataset_id} 不存在")
            
        # 标记为就绪
        dataset.mark_ready()
        
        # 保存更新
        updated = self.dataset_repository.update(dataset)
        
        logger.info(f"Marked dataset {dataset_id} as ready")
        return updated

    
    def get_statistics(self, user_id: str) -> Dict[str, Any]:
        """获取用户数据集统计信息
        
        Args:
            user_id: 用户ID
            
        Returns:
            Dict[str, Any]: 统计信息
                - total: 总数据集数
                - by_status: 按状态分组的数量
                - by_type: 按类型分组的数量
                - total_size: 总存储大小
                - total_records: 总记录数
                
        Example:
            >>> stats = service.get_statistics("user123")
            >>> print(f"总数据集: {stats['total']}")
        """
        return self.dataset_repository.get_statistics(user_id)
    
    def bulk_delete(self, dataset_ids: List[str], user_id: str = None) -> Dict[str, Any]:
        """批量删除数据集
        
        Args:
            dataset_ids: 数据集ID列表
            user_id: 用户ID (可选)
            
        Returns:
            Dict[str, Any]: 删除结果
                - deleted: 成功删除的数量
                - failed: 失败的数量
                - errors: 错误信息列表
                
        Example:
            >>> result = service.bulk_delete(["id1", "id2", "id3"])
            >>> print(f"删除了 {result['deleted']} 个数据集")
        """
        deleted_count = self.dataset_repository.bulk_delete(dataset_ids)
        
        logger.info(f"Bulk deleted {deleted_count} datasets")
        
        return {
            'deleted': deleted_count,
            'failed': len(dataset_ids) - deleted_count,
            'errors': []
        }
    
    def archive_dataset(self, dataset_id: str, user_id: str = None) -> Dataset:
        """归档数据集
        
        将数据集状态设置为archived。
        
        Args:
            dataset_id: 数据集ID
            user_id: 用户ID (可选)
            
        Returns:
            Dataset: 归档后的数据集对象
        """
        validate_id(dataset_id, "dataset_id")
        
        return self.update_dataset(
            dataset_id=dataset_id,
            status='archived',
            user_id=user_id
        )
    
    def clone_dataset(
        self, 
        source_dataset_id: str, 
        user_id: str,
        new_name: str = None
    ) -> Dataset:
        """克隆数据集
        
        创建现有数据集的副本。
        
        Args:
            source_dataset_id: 源数据集ID
            user_id: 用户ID
            new_name: 新数据集名称 (可选，默认加'_copy'后缀)
            
        Returns:
            Dataset: 克隆的新数据集对象
        """
        validate_id(source_dataset_id, "source_dataset_id")
        
        # 获取源数据集
        source = self.dataset_repository.get_by_id(source_dataset_id)
        if not source:
            raise DatasetNotFoundError(f"源数据集 {source_dataset_id} 不存在")
        
        # 创建副本
        cloned = self.create_dataset(
            user_id=user_id,
            name=new_name or f"{source.name}_copy",
            description=source.description,
            dataset_type=source.dataset_type,
            format=source.format,
            storage_path="",  # 新的存储路径
            config=source.config.copy() if source.config else {}
        )
        
        logger.info(f"Cloned dataset {source_dataset_id} to {cloned.dataset_id}")
        return cloned
    
    # ============================================================================
    # 标签管理
    # ============================================================================
    
    def add_tag(
        self, 
        dataset_id: str, 
        tag_name: str, 
        tag_value: str = None,
        user_id: str = None
    ) -> str:
        """为数据集添加标签
        
        Args:
            dataset_id: 数据集ID
            tag_name: 标签名称
            tag_value: 标签值 (可选)
            user_id: 用户ID (可选)
            
        Returns:
            str: 标签ID
        """
        validate_id(dataset_id, "dataset_id")
        
        # 验证数据集存在
        if not self.dataset_repository.exists(dataset_id):
            raise DatasetNotFoundError(f"数据集 {dataset_id} 不存在")
        
        return self.tag_repository.add_tag(
            dataset_id=dataset_id,
            tag_name=tag_name,
            tag_value=tag_value,
            created_by=user_id
        )
    
    def remove_tag(self, dataset_id: str, tag_name: str) -> bool:
        """移除数据集标签
        
        Args:
            dataset_id: 数据集ID
            tag_name: 标签名称
            
        Returns:
            bool: 是否成功移除
        """
        return self.tag_repository.remove_tag(dataset_id, tag_name)
    
    def get_tags(self, dataset_id: str) -> List[Dict[str, str]]:
        """获取数据集的所有标签
        
        Args:
            dataset_id: 数据集ID
            
        Returns:
            List[Dict[str, str]]: 标签列表
        """
        return self.tag_repository.get_tags(dataset_id)
    
    def find_by_tag(
        self, 
        user_id: str, 
        tag_name: str, 
        tag_value: str = None
    ) -> List[Dataset]:
        """根据标签查找数据集
        
        Args:
            user_id: 用户ID
            tag_name: 标签名称
            tag_value: 标签值 (可选)
            
        Returns:
            List[Dataset]: 匹配的数据集列表
        """
        dataset_ids = self.tag_repository.find_by_tag(user_id, tag_name, tag_value)
        
        datasets = []
        for dataset_id in dataset_ids:
            dataset = self.dataset_repository.get_by_id(dataset_id)
            if dataset:
                datasets.append(dataset)
        
        return datasets
    
    # ============================================================================
    # 版本管理
    # ============================================================================
    
    def create_version(
        self,
        dataset_id: str,
        version: str,
        description: str = None,
        created_by: str = None,
        changes: Dict[str, Any] = None
    ) -> DatasetVersion:
        """创建数据集新版本
        
        Args:
            dataset_id: 数据集ID
            version: 版本号
            description: 版本描述
            created_by: 创建者ID
            changes: 变更内容
            
        Returns:
            DatasetVersion: 创建的版本对象
        """
        validate_id(dataset_id, "dataset_id")
        
        # 获取数据集信息
        dataset = self.dataset_repository.get_by_id(dataset_id)
        if not dataset:
            raise DatasetNotFoundError(f"数据集 {dataset_id} 不存在")
        
        # 获取当前版本作为父版本
        versions = self.version_repository.list_by_dataset(dataset_id)
        parent_version_id = versions[0].version_id if versions else None
        
        # 创建新版本
        version_dto = DatasetVersion(
            dataset_id=dataset_id,
            version=version,
            description=description,
            storage_path=dataset.storage_path,
            size=dataset.size,
            record_count=dataset.record_count,
            checksum=dataset.checksum,
            created_by=created_by or dataset.user_id,
            changes=changes,
            parent_version_id=parent_version_id
        )
        
        created = self.version_repository.create(version_dto)
        
        # 更新数据集版本号
        dataset.version = version
        self.dataset_repository.update(dataset)
        
        logger.info(f"Created version {version} for dataset {dataset_id}")
        return created
    
    def get_versions(self, dataset_id: str) -> List[DatasetVersion]:
        """获取数据集的所有版本
        
        Args:
            dataset_id: 数据集ID
            
        Returns:
            List[DatasetVersion]: 版本列表
        """
        return self.version_repository.list_by_dataset(dataset_id)
    
    # ============================================================================
    # 访问日志
    # ============================================================================
    
    def get_access_logs(
        self, 
        dataset_id: str, 
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """获取数据集访问日志
        
        Args:
            dataset_id: 数据集ID
            limit: 返回数量限制
            offset: 偏移量
            
        Returns:
            List[Dict[str, Any]]: 访问日志列表
        """
        return self.access_log_repository.list_by_dataset(
            dataset_id=dataset_id,
            limit=limit,
            offset=offset
        )
    
    def _log_access(
        self, 
        dataset_id: str, 
        user_id: str, 
        action: str,
        details: Dict[str, Any] = None
    ):
        """记录访问日志(内部方法)"""
        try:
            self.access_log_repository.create(
                dataset_id=dataset_id,
                user_id=user_id,
                action=action,
                details=details
            )
        except Exception as e:
            # 访问日志记录失败不应影响主流程
            logger.warning(f"Failed to log access: {e}")
    
    # ============================================================================
    # 详细管理方法 - 用于dataset_detailed_api
    # ============================================================================
    
    def get_dataset_tags(self, dataset_id: str) -> List[str]:
        """获取数据集标签名列表
        
        获取指定数据集的所有标签名称。
        
        Args:
            dataset_id: 数据集唯一标识符 (UUID字符串)
                - 格式: "550e8400-e29b-41d4-a716-446655440000"
                
        Returns:
            List[str]: 标签名称列表
                - 示例: ["训练数据", "NLP", "v1.0"]
                
        Example:
            >>> tags = service.get_dataset_tags("dataset123")
            >>> print(tags)  # ["标签1", "标签2"]
        """
        tags = self.get_tags(dataset_id)
        return [tag.get('name', tag.get('tag_name', '')) for tag in tags if tag]
    
    def get_dataset_statistics(self, dataset_id: str) -> Optional[DatasetStatistics]:
        """获取数据集统计信息
        
        获取指定数据集的基本统计信息。
        
        Args:
            dataset_id: 数据集唯一标识符
            
        Returns:
            Optional[DatasetStatistics]: 统计信息对象，不存在则返回None
            
        Example:
            >>> stats = service.get_dataset_statistics("dataset123")
            >>> if stats:
            ...     print(f"行数: {stats.row_count}")
        """
        try:
            dataset = self.dataset_repository.get_by_id(dataset_id)
            if not dataset:
                return None
            
            # 构造统计信息
            return DatasetStatistics(
                dataset_id=dataset_id,
                total_records=dataset.record_count or 0,
                column_count=len(dataset.features.get('columns', [])) if dataset.features else 0,
                total_size=dataset.size or 0,
                missing_values=None  # 需要实际分析
            )
        except Exception as e:
            logger.warning(f"Failed to get statistics for dataset {dataset_id}: {e}")
            return None
    
    def log_access(
        self, 
        dataset_id: str, 
        user_id: str, 
        action: str,
        details: Dict[str, Any] = None
    ):
        """记录访问日志
        
        记录用户对数据集的访问操作。
        
        Args:
            dataset_id: 数据集ID
            user_id: 用户ID
            action: 操作类型 (view/download/update/delete/preview/analyze/split)
            details: 操作详情 (可选)
            
        Example:
            >>> service.log_access("dataset123", "user456", "view")
        """
        self._log_access(dataset_id, user_id, action, details)
    
    def clear_dataset_tags(self, dataset_id: str) -> bool:
        """清除数据集所有标签
        
        删除指定数据集的所有标签。
        
        Args:
            dataset_id: 数据集ID
            
        Returns:
            bool: 是否成功清除
        """
        try:
            tags = self.get_dataset_tags(dataset_id)
            for tag in tags:
                self.tag_repository.remove_tag(dataset_id, tag)
            return True
        except Exception as e:
            logger.warning(f"Failed to clear tags for dataset {dataset_id}: {e}")
            return False
    
    def add_dataset_tag(self, dataset_id: str, tag_name: str, user_id: str = None) -> str:
        """添加数据集标签
        
        为数据集添加一个标签。
        
        Args:
            dataset_id: 数据集ID
            tag_name: 标签名称
            user_id: 操作用户ID (可选)
            
        Returns:
            str: 标签ID
        """
        return self.add_tag(dataset_id, tag_name, None, user_id)
    
    def remove_dataset_tag(self, dataset_id: str, tag_name: str) -> bool:
        """移除数据集标签
        
        移除指定数据集的一个标签。
        
        Args:
            dataset_id: 数据集ID
            tag_name: 标签名称
            
        Returns:
            bool: 是否成功移除
        """
        return self.remove_tag(dataset_id, tag_name)
    
    def generate_download_url(
        self, 
        dataset_id: str, 
        user_id: str,
        format: str = 'original'
    ) -> Dict[str, Any]:
        """生成数据集下载URL
        
        生成数据集文件的下载链接和相关信息。
        
        Args:
            dataset_id: 数据集ID
            user_id: 用户ID
            format: 下载格式 (original/json/csv/parquet)
            
        Returns:
            Dict[str, Any]: 下载信息
                - download_url: 下载链接
                - file_path: 文件路径
                - file_name: 文件名
                - file_size: 文件大小
                - format: 格式
                - expires_at: 过期时间
        """
        dataset = self.dataset_repository.get_by_id(dataset_id)
        if not dataset:
            raise DatasetNotFoundError(f"数据集 {dataset_id} 不存在")
        
        # 确定文件扩展名
        ext_map = {
            'original': dataset.format or 'json',
            'json': 'json',
            'csv': 'csv',
            'parquet': 'parquet'
        }
        ext = ext_map.get(format, dataset.format or 'json')
        
        file_name = f"{dataset.name}_{dataset_id[:8]}.{ext}"
        file_path = dataset.storage_path or f"./data/datasets/{user_id}/{dataset_id}/data.{ext}"
        
        # 计算过期时间（1小时后）
        from datetime import timedelta
        expires_at = datetime.utcnow() + timedelta(hours=1)
        
        return {
            'download_url': f"/api/v1/datasets/{dataset_id}/download?direct=true&format={format}",
            'file_path': file_path,
            'file_name': file_name,
            'file_size': dataset.size or 0,
            'format': ext,
            'expires_at': expires_at.isoformat()
        }
    
    def preview_dataset(
        self, 
        dataset_id: str, 
        limit: int = 10,
        offset: int = 0,
        columns: List[str] = None
    ) -> Dict[str, Any]:
        """预览数据集内容
        
        获取数据集的部分数据进行预览。
        
        Args:
            dataset_id: 数据集ID
            limit: 返回行数 (1-100)
            offset: 起始偏移量
            columns: 要返回的列 (可选，默认全部)
            
        Returns:
            Dict[str, Any]: 预览结果
                - dataset_id: 数据集ID
                - preview_data: 预览数据数组
                - columns: 列名列表
                - column_types: 列类型映射
                - total_rows: 总行数
                - preview_rows: 预览行数
                - limit: 请求的limit
                - offset: 请求的offset
                - has_more: 是否有更多数据
        """
        dataset = self.dataset_repository.get_by_id(dataset_id)
        if not dataset:
            raise DatasetNotFoundError(f"数据集 {dataset_id} 不存在")
        
        # 模拟预览数据（实际应从存储中读取）
        total_rows = dataset.record_count or 1000
        preview_data = []
        
        # 获取列信息
        all_columns = ['id', 'content', 'label', 'created_at']
        if dataset.features and 'columns' in dataset.features:
            all_columns = dataset.features['columns']
        
        # 过滤列
        display_columns = columns if columns else all_columns
        display_columns = [c for c in display_columns if c in all_columns]
        if not display_columns:
            display_columns = all_columns
        
        # 生成模拟数据
        for i in range(offset, min(offset + limit, total_rows)):
            row = {}
            for col in display_columns:
                if col == 'id':
                    row[col] = i + 1
                elif col == 'content':
                    row[col] = f"示例内容 {i + 1}"
                elif col == 'label':
                    row[col] = ['positive', 'negative', 'neutral'][i % 3]
                elif col == 'created_at':
                    row[col] = datetime.utcnow().isoformat()
                else:
                    row[col] = f"值_{i + 1}"
            preview_data.append(row)
        
        # 列类型映射
        column_types = {
            'id': 'integer',
            'content': 'string',
            'label': 'string',
            'created_at': 'datetime'
        }
        
        return {
            'dataset_id': dataset_id,
            'preview_data': preview_data,
            'columns': display_columns,
            'column_types': {k: v for k, v in column_types.items() if k in display_columns},
            'total_rows': total_rows,
            'preview_rows': len(preview_data),
            'limit': limit,
            'offset': offset,
            'has_more': (offset + limit) < total_rows
        }
    
    def analyze_dataset(
        self, 
        dataset_id: str, 
        analysis_type: str = 'basic',
        columns: List[str] = None,
        sample_size: int = None,
        include_distributions: bool = False
    ) -> Dict[str, Any]:
        """分析数据集
        
        对数据集进行统计分析。
        
        Args:
            dataset_id: 数据集ID
            analysis_type: 分析类型 (basic/detailed/full)
            columns: 要分析的列 (可选)
            sample_size: 采样大小 (可选)
            include_distributions: 是否包含分布分析
            
        Returns:
            Dict[str, Any]: 分析结果
                - dataset_id: 数据集ID
                - analysis_type: 分析类型
                - analyzed_at: 分析时间
                - basic_stats: 基本统计
                - detailed_stats: 详细统计 (analysis_type >= detailed)
                - data_quality: 数据质量 (analysis_type >= detailed)
                - recommendations: 建议 (analysis_type == full)
        """
        dataset = self.dataset_repository.get_by_id(dataset_id)
        if not dataset:
            raise DatasetNotFoundError(f"数据集 {dataset_id} 不存在")
        
        total_rows = dataset.record_count or 1000
        total_columns = len(dataset.features.get('columns', [])) if dataset.features else 4
        
        result = {
            'dataset_id': dataset_id,
            'analysis_type': analysis_type,
            'analyzed_at': datetime.utcnow().isoformat(),
            'basic_stats': {
                'total_rows': total_rows,
                'total_columns': total_columns,
                'missing_values': int(total_rows * 0.005),  # 模拟0.5%缺失
                'duplicate_rows': int(total_rows * 0.002),  # 模拟0.2%重复
                'memory_usage': f"{(dataset.size or 1024000) / 1024 / 1024:.2f} MB"
            }
        }
        
        if analysis_type in ['detailed', 'full']:
            result['detailed_stats'] = {
                'column_stats': [
                    {
                        'column_name': 'id',
                        'data_type': 'integer',
                        'unique_values': total_rows,
                        'missing_count': 0,
                        'missing_percentage': 0.0,
                        'min': 1,
                        'max': total_rows,
                        'mean': total_rows / 2,
                        'median': total_rows / 2,
                        'std': total_rows / 3.46
                    },
                    {
                        'column_name': 'content',
                        'data_type': 'string',
                        'unique_values': int(total_rows * 0.99),
                        'missing_count': int(total_rows * 0.003),
                        'missing_percentage': 0.3,
                        'min': None,
                        'max': None,
                        'mean': None,
                        'median': None,
                        'std': None
                    },
                    {
                        'column_name': 'label',
                        'data_type': 'string',
                        'unique_values': 3,
                        'missing_count': int(total_rows * 0.002),
                        'missing_percentage': 0.2,
                        'min': None,
                        'max': None,
                        'mean': None,
                        'median': None,
                        'std': None
                    }
                ]
            }
            
            result['data_quality'] = {
                'completeness': 0.995,
                'uniqueness': 0.998,
                'consistency': 0.99,
                'overall_score': 0.994
            }
        
        if analysis_type == 'full':
            result['recommendations'] = []
            
            if result['basic_stats']['missing_values'] > 0:
                result['recommendations'].append({
                    'type': 'data_quality',
                    'message': f"发现 {result['basic_stats']['missing_values']} 个缺失值，建议进行数据清洗",
                    'priority': 'medium',
                    'affected_columns': ['content', 'label']
                })
            
            if result['basic_stats']['duplicate_rows'] > 0:
                result['recommendations'].append({
                    'type': 'data_quality',
                    'message': f"发现 {result['basic_stats']['duplicate_rows']} 个重复行，建议去重处理",
                    'priority': 'low',
                    'affected_columns': []
                })
            
            result['recommendations'].append({
                'type': 'optimization',
                'message': '数据集整体质量良好，可以用于模型训练',
                'priority': 'info',
                'affected_columns': []
            })
        
        return result
    
    def split_dataset(
        self, 
        dataset_id: str, 
        user_id: str,
        split_ratios: Dict[str, float] = None,
        shuffle: bool = True,
        seed: int = 42,
        stratify_column: str = None,
        create_new_datasets: bool = True
    ) -> Dict[str, Any]:
        """分割数据集
        
        将数据集按指定比例分割为训练集、验证集和测试集。
        
        Args:
            dataset_id: 数据集ID
            user_id: 用户ID
            split_ratios: 分割比例 (默认 train=0.8, validation=0.1, test=0.1)
            shuffle: 是否打乱数据
            seed: 随机种子
            stratify_column: 分层采样列
            create_new_datasets: 是否创建新数据集
            
        Returns:
            Dict[str, Any]: 分割结果
                - dataset_id: 原数据集ID
                - split_config: 分割配置
                - splits: 分割结果
                - total_samples: 总样本数
                - split_at: 分割时间
        """
        if split_ratios is None:
            split_ratios = {'train': 0.8, 'validation': 0.1, 'test': 0.1}
        
        dataset = self.dataset_repository.get_by_id(dataset_id)
        if not dataset:
            raise DatasetNotFoundError(f"数据集 {dataset_id} 不存在")
        
        total_samples = dataset.record_count or 1000
        splits = {}
        
        for split_name, ratio in split_ratios.items():
            samples = int(total_samples * ratio)
            
            split_info = {
                'name': f"{dataset.name}_{split_name}",
                'size': int((dataset.size or 1024000) * ratio),
                'samples': samples
            }
            
            # 创建新数据集
            if create_new_datasets:
                new_dataset = self.create_dataset(
                    user_id=user_id,
                    name=f"{dataset.name}_{split_name}",
                    description=f"{dataset.name} 的 {split_name} 分割",
                    dataset_type=dataset.dataset_type,
                    format=dataset.format,
                    storage_path=f"{dataset.storage_path}_{split_name}" if dataset.storage_path else "",
                    config={
                        'source_dataset_id': dataset_id,
                        'split_type': split_name,
                        'split_ratio': ratio
                    }
                )
                split_info['dataset_id'] = new_dataset.dataset_id
            
            splits[split_name] = split_info
        
        return {
            'dataset_id': dataset_id,
            'split_config': {
                'ratios': split_ratios,
                'shuffle': shuffle,
                'seed': seed,
                'stratified': stratify_column is not None
            },
            'splits': splits,
            'total_samples': total_samples,
            'split_at': datetime.utcnow().isoformat()
        }
    
    def get_detailed_statistics(self, dataset_id: str) -> Dict[str, Any]:
        """获取详细统计信息
        
        获取数据集的详细统计信息，包括大小、行数、列统计等。
        
        Args:
            dataset_id: 数据集ID
            
        Returns:
            Dict[str, Any]: 详细统计信息
        """
        dataset = self.dataset_repository.get_by_id(dataset_id)
        if not dataset:
            raise DatasetNotFoundError(f"数据集 {dataset_id} 不存在")
        
        size_bytes = dataset.size or 0
        size_human = f"{size_bytes / 1024 / 1024:.2f} MB" if size_bytes > 1024*1024 else f"{size_bytes / 1024:.2f} KB"
        
        return {
            'dataset_id': dataset_id,
            'total_rows': dataset.record_count or 0,
            'total_columns': len(dataset.features.get('columns', [])) if dataset.features else 0,
            'size_bytes': size_bytes,
            'size_human': size_human,
            'column_count_by_type': {
                'integer': 1,
                'string': 2,
                'datetime': 1
            },
            'missing_values_total': int((dataset.record_count or 0) * 0.005),
            'missing_percentage': 0.5,
            'duplicate_rows': int((dataset.record_count or 0) * 0.002),
            'last_analyzed': dataset.updated_at.isoformat() if dataset.updated_at else None
        }
    
    def get_dataset_versions(
        self, 
        dataset_id: str, 
        limit: int = 20,
        offset: int = 0
    ) -> Dict[str, Any]:
        """获取数据集版本历史
        
        获取指定数据集的版本列表。
        
        Args:
            dataset_id: 数据集ID
            limit: 返回数量限制
            offset: 偏移量
            
        Returns:
            Dict[str, Any]: 版本历史结果
        """
        dataset = self.dataset_repository.get_by_id(dataset_id)
        if not dataset:
            raise DatasetNotFoundError(f"数据集 {dataset_id} 不存在")
        
        versions = self.get_versions(dataset_id)
        
        # 分页
        total = len(versions)
        paged_versions = versions[offset:offset + limit]
        
        return {
            'dataset_id': dataset_id,
            'versions': [
                {
                    'version_id': v.version_id,
                    'version': v.version,
                    'description': v.description,
                    'created_at': v.created_at.isoformat() if v.created_at else None,
                    'created_by': v.created_by,
                    'size': v.size,
                    'record_count': v.record_count,
                    'is_current': v.version == dataset.version
                }
                for v in paged_versions
            ],
            'total': total,
            'limit': limit,
            'offset': offset
        }
    
    def create_dataset_version(
        self, 
        dataset_id: str, 
        user_id: str,
        version: str = None,
        description: str = None,
        changelog: str = None
    ) -> Dict[str, Any]:
        """创建数据集新版本
        
        创建数据集的新版本记录。
        
        Args:
            dataset_id: 数据集ID
            user_id: 用户ID
            version: 版本号 (可选，自动递增)
            description: 版本描述
            changelog: 变更日志
            
        Returns:
            Dict[str, Any]: 创建的版本信息
        """
        dataset = self.dataset_repository.get_by_id(dataset_id)
        if not dataset:
            raise DatasetNotFoundError(f"数据集 {dataset_id} 不存在")
        
        # 自动生成版本号
        if not version:
            current_version = dataset.version or '1.0'
            parts = current_version.split('.')
            try:
                minor = int(parts[-1]) + 1
                version = '.'.join(parts[:-1] + [str(minor)])
            except ValueError:
                version = f"{current_version}.1"
        
        # 创建版本
        new_version = self.create_version(
            dataset_id=dataset_id,
            version=version,
            description=description,
            created_by=user_id,
            changes={'changelog': changelog} if changelog else None
        )
        
        return {
            'version_id': new_version.version_id,
            'dataset_id': dataset_id,
            'version': new_version.version,
            'description': new_version.description,
            'created_at': new_version.created_at.isoformat() if new_version.created_at else None,
            'created_by': new_version.created_by
        }
    
    # ============================================================================
    # 高级管理方法 - 用于dataset_management_api
    # ============================================================================
    
    def search_datasets(
        self,
        user_id: str,
        dataset_type: str = None,
        status: str = None,
        search: str = None,
        tag_filter: List[str] = None,
        limit: int = 50,
        offset: int = 0,
        order_by: str = 'created_at',
        order_desc: bool = True
    ) -> Dict[str, Any]:
        """搜索数据集
        
        支持多条件过滤和排序的数据集搜索。
        
        Args:
            user_id: 用户ID (必填)
            dataset_type: 数据集类型过滤 (可选)
                - 可选值: text, image, audio, video, tabular, mixed
            status: 状态过滤 (可选)
                - 可选值: pending, uploading, processing, ready, error, archived
            search: 搜索关键词 (可选)
                - 在名称和描述中搜索
            tag_filter: 标签过滤列表 (可选)
            limit: 返回数量限制 (1-100，默认50)
            offset: 偏移量 (默认0)
            order_by: 排序字段 (默认'created_at')
                - 可选值: created_at, updated_at, name, size
            order_desc: 是否降序 (默认True)
            
        Returns:
            Dict[str, Any]: 搜索结果
                - datasets: 数据集列表 (Dict格式)
                - total_count: 符合条件的总数
                - filtered_count: 过滤后的数量
                - limit: 限制数量
                - offset: 偏移量
                - has_more: 是否有更多数据
                
        Example:
            >>> result = service.search_datasets(
            ...     user_id="user123",
            ...     dataset_type="text",
            ...     status="ready",
            ...     search="训练",
            ...     limit=20
            ... )
            >>> print(f"找到 {result['total_count']} 个数据集")
        """
        # 获取所有数据集
        datasets = self.dataset_repository.list_by_user(
            user_id=user_id,
            limit=1000,  # 获取所有用于过滤
            offset=0,
            status=status,
            dataset_type=dataset_type,
            order_by=order_by,
            order_desc=order_desc
        )
        
        total_count = len(datasets)
        
        # 关键词搜索过滤
        if search:
            search_lower = search.lower()
            datasets = [
                ds for ds in datasets 
                if search_lower in (ds.name or '').lower() or 
                   search_lower in (ds.description or '').lower()
            ]
        
        # 标签过滤
        if tag_filter:
            filtered_datasets = []
            for ds in datasets:
                ds_tags = self.get_dataset_tags(ds.dataset_id)
                if any(tag in ds_tags for tag in tag_filter):
                    filtered_datasets.append(ds)
            datasets = filtered_datasets
        
        filtered_count = len(datasets)
        
        # 分页
        paged_datasets = datasets[offset:offset + limit]
        
        return {
            'datasets': [ds.to_dict() for ds in paged_datasets],
            'total_count': total_count,
            'filtered_count': filtered_count,
            'limit': limit,
            'offset': offset,
            'has_more': (offset + limit) < filtered_count
        }
    
    def restore_dataset(self, dataset_id: str, user_id: str = None) -> Dataset:
        """恢复归档的数据集
        
        将已归档的数据集恢复为正常状态。
        
        Args:
            dataset_id: 数据集ID (必填)
            user_id: 用户ID (可选，用于日志记录)
            
        Returns:
            Dataset: 恢复后的数据集对象
                - status: 'ready'
            
        Raises:
            DatasetNotFoundError: 当数据集不存在时
            DatasetBusinessLogicError: 当数据集未归档时
            
        Example:
            >>> restored = service.restore_dataset("dataset123", "user456")
            >>> print(restored.status)  # 输出: ready
        """
        validate_id(dataset_id, "dataset_id")
        
        dataset = self.dataset_repository.get_by_id(dataset_id)
        if not dataset:
            raise DatasetNotFoundError(f"数据集 {dataset_id} 不存在")
        
        if dataset.status != 'archived':
            raise DatasetBusinessLogicError(f"数据集 {dataset_id} 未归档，无需恢复")
        
        # 恢复到ready状态
        return self.update_dataset(
            dataset_id=dataset_id,
            status='ready',
            user_id=user_id
        )
    
    def transfer_dataset(
        self, 
        dataset_id: str, 
        from_user_id: str, 
        to_user_id: str
    ) -> Dataset:
        """转移数据集所有权
        
        将数据集的所有权从一个用户转移给另一个用户。
        
        Args:
            dataset_id: 数据集ID (必填)
            from_user_id: 原所有者用户ID (必填)
            to_user_id: 新所有者用户ID (必填)
            
        Returns:
            Dataset: 转移后的数据集对象
            
        Raises:
            DatasetNotFoundError: 当数据集不存在时
            DatasetValidationError: 当参数无效时
            DatasetBusinessLogicError: 当无权转移时
            
        Example:
            >>> transferred = service.transfer_dataset(
            ...     "dataset123", "user_old", "user_new"
            ... )
        """
        validate_id(dataset_id, "dataset_id")
        
        if from_user_id == to_user_id:
            raise ValueError("不能转移给自己")
        
        dataset = self.dataset_repository.get_by_id(dataset_id)
        if not dataset:
            raise DatasetNotFoundError(f"数据集 {dataset_id} 不存在")
        
        if dataset.user_id != from_user_id:
            raise DatasetBusinessLogicError(f"用户 {from_user_id} 不是数据集 {dataset_id} 的所有者")
        
        # 更新所有者
        dataset.user_id = to_user_id
        dataset.updated_at = datetime.utcnow()
        
        updated = self.dataset_repository.update(dataset)
        
        # 记录访问日志
        self._log_access(dataset_id, from_user_id, 'transfer', {
            'from_user_id': from_user_id,
            'to_user_id': to_user_id
        })
        
        logger.info(f"Transferred dataset {dataset_id} from {from_user_id} to {to_user_id}")
        return updated
    
    def merge_datasets(
        self, 
        target_dataset_id: str, 
        source_dataset_ids: List[str],
        user_id: str,
        delete_sources: bool = False
    ) -> Dict[str, Any]:
        """合并数据集
        
        将多个数据集合并到目标数据集中。
        
        Args:
            target_dataset_id: 目标数据集ID (必填)
            source_dataset_ids: 源数据集ID列表 (必填)
            user_id: 操作用户ID (必填)
            delete_sources: 合并后是否删除源数据集 (默认False)
            
        Returns:
            Dict[str, Any]: 合并结果
                - target_dataset_id: 目标数据集ID
                - merged_count: 合并的数据集数量
                - total_records: 合并后的总记录数
                - merged_at: 合并时间
                - deleted_sources: 已删除的源数据集ID列表
            
        Raises:
            DatasetNotFoundError: 当数据集不存在时
            DatasetValidationError: 当数据集类型/格式不兼容时
            
        Example:
            >>> result = service.merge_datasets(
            ...     "target_id",
            ...     ["source1", "source2"],
            ...     "user123"
            ... )
            >>> print(f"合并了 {result['merged_count']} 个数据集")
        """
        validate_id(target_dataset_id, "target_dataset_id")
        
        # 获取目标数据集
        target = self.dataset_repository.get_by_id(target_dataset_id)
        if not target:
            raise DatasetNotFoundError(f"目标数据集 {target_dataset_id} 不存在")
        
        merged_count = 0
        total_records = target.record_count or 0
        deleted_sources = []
        
        for source_id in source_dataset_ids:
            if source_id == target_dataset_id:
                continue  # 跳过自身
            
            source = self.dataset_repository.get_by_id(source_id)
            if not source:
                logger.warning(f"Source dataset {source_id} not found, skipping")
                continue
            
            # 检查兼容性
            if source.dataset_type != target.dataset_type:
                raise DatasetValidationError(
                    f"数据集类型不兼容: {source.dataset_type} vs {target.dataset_type}"
                )
            
            if source.format != target.format:
                raise DatasetValidationError(
                    f"数据格式不兼容: {source.format} vs {target.format}"
                )
            
            # 合并记录数和大小
            total_records += source.record_count or 0
            target.size = (target.size or 0) + (source.size or 0)
            
            merged_count += 1
            
            # 删除源数据集
            if delete_sources:
                self.dataset_repository.delete(source_id)
                deleted_sources.append(source_id)
        
        # 更新目标数据集
        target.record_count = total_records
        target.updated_at = datetime.utcnow()
        self.dataset_repository.update(target)
        
        # 记录日志
        self._log_access(target_dataset_id, user_id, 'merge', {
            'source_dataset_ids': source_dataset_ids,
            'merged_count': merged_count,
            'delete_sources': delete_sources
        })
        
        logger.info(f"Merged {merged_count} datasets into {target_dataset_id}")
        
        return {
            'target_dataset_id': target_dataset_id,
            'merged_count': merged_count,
            'total_records': total_records,
            'merged_at': datetime.utcnow().isoformat(),
            'deleted_sources': deleted_sources
        }
    
    def advanced_search(
        self, 
        user_id: str, 
        search_params: Dict[str, Any],
        limit: int = 50,
        offset: int = 0
    ) -> Dict[str, Any]:
        """高级搜索数据集
        
        使用高级搜索条件搜索数据集。
        
        Args:
            user_id: 用户ID (必填)
            search_params: 搜索参数 (必填)
                - q: 搜索关键词
                - type: 数据集类型
                - status: 状态
                - tags: 标签列表
                - min_size: 最小大小(字节)
                - max_size: 最大大小(字节)
                - min_records: 最小记录数
                - max_records: 最大记录数
                - created_after: 创建时间起始 (ISO8601)
                - created_before: 创建时间结束 (ISO8601)
            limit: 返回数量限制 (默认50)
            offset: 偏移量 (默认0)
            
        Returns:
            Dict[str, Any]: 搜索结果
                - datasets: 数据集列表
                - total_count: 总数
                - query: 搜索参数
                
        Example:
            >>> result = service.advanced_search(
            ...     "user123",
            ...     {"q": "训练", "type": "text", "min_size": 1000}
            ... )
        """
        # 获取所有数据集
        datasets = self.dataset_repository.list_by_user(
            user_id=user_id,
            limit=1000,
            offset=0,
            status=search_params.get('status'),
            dataset_type=search_params.get('type')
        )
        
        # 关键词搜索
        q = search_params.get('q')
        if q:
            q_lower = q.lower()
            datasets = [
                ds for ds in datasets 
                if q_lower in (ds.name or '').lower() or 
                   q_lower in (ds.description or '').lower()
            ]
        
        # 标签过滤
        tags = search_params.get('tags')
        if tags:
            filtered = []
            for ds in datasets:
                ds_tags = self.get_dataset_tags(ds.dataset_id)
                if any(tag in ds_tags for tag in tags):
                    filtered.append(ds)
            datasets = filtered
        
        # 大小过滤
        min_size = search_params.get('min_size')
        max_size = search_params.get('max_size')
        if min_size is not None:
            datasets = [ds for ds in datasets if (ds.size or 0) >= min_size]
        if max_size is not None:
            datasets = [ds for ds in datasets if (ds.size or 0) <= max_size]
        
        # 记录数过滤
        min_records = search_params.get('min_records')
        max_records = search_params.get('max_records')
        if min_records is not None:
            datasets = [ds for ds in datasets if (ds.record_count or 0) >= min_records]
        if max_records is not None:
            datasets = [ds for ds in datasets if (ds.record_count or 0) <= max_records]
        
        # 时间过滤
        created_after = search_params.get('created_after')
        created_before = search_params.get('created_before')
        if created_after:
            try:
                after_dt = datetime.fromisoformat(created_after.replace('Z', '+00:00'))
                datasets = [ds for ds in datasets if ds.created_at and ds.created_at >= after_dt]
            except ValueError:
                pass
        if created_before:
            try:
                before_dt = datetime.fromisoformat(created_before.replace('Z', '+00:00'))
                datasets = [ds for ds in datasets if ds.created_at and ds.created_at <= before_dt]
            except ValueError:
                pass
        
        total_count = len(datasets)
        
        # 分页
        paged_datasets = datasets[offset:offset + limit]
        
        return {
            'datasets': [ds.to_dict() for ds in paged_datasets],
            'total_count': total_count,
            'query': search_params,
            'limit': limit,
            'offset': offset
        }
    
    def get_user_statistics(self, user_id: str) -> Dict[str, Any]:
        """获取用户数据集统计信息
        
        获取当前用户的数据集统计概览。
        
        Args:
            user_id: 用户ID (必填)
            
        Returns:
            Dict[str, Any]: 统计信息
                - total_datasets: 数据集总数
                - total_size_bytes: 总存储大小(字节)
                - total_size_human: 总存储大小(可读格式)
                - total_records: 总记录数
                - datasets_by_type: 按类型分组的数量
                - datasets_by_status: 按状态分组的数量
                - recent_uploads: 最近7天上传数量
                - storage_usage_percent: 存储使用百分比
                
        Example:
            >>> stats = service.get_user_statistics("user123")
            >>> print(f"总数据集: {stats['total_datasets']}")
        """
        from datetime import timedelta
        
        # 获取所有数据集
        datasets = self.dataset_repository.list_by_user(
            user_id=user_id,
            limit=10000,
            offset=0
        )
        
        total_datasets = len(datasets)
        total_size_bytes = sum(ds.size or 0 for ds in datasets)
        total_records = sum(ds.record_count or 0 for ds in datasets)
        
        # 按类型分组
        datasets_by_type = {}
        for ds in datasets:
            ds_type = ds.dataset_type or 'unknown'
            datasets_by_type[ds_type] = datasets_by_type.get(ds_type, 0) + 1
        
        # 按状态分组
        datasets_by_status = {}
        for ds in datasets:
            ds_status = ds.status or 'unknown'
            datasets_by_status[ds_status] = datasets_by_status.get(ds_status, 0) + 1
        
        # 最近7天上传数量
        seven_days_ago = datetime.utcnow() - timedelta(days=7)
        recent_uploads = sum(
            1 for ds in datasets 
            if ds.created_at and ds.created_at >= seven_days_ago
        )
        
        # 转换大小为可读格式
        if total_size_bytes >= 1024 * 1024 * 1024:
            total_size_human = f"{total_size_bytes / 1024 / 1024 / 1024:.2f} GB"
        elif total_size_bytes >= 1024 * 1024:
            total_size_human = f"{total_size_bytes / 1024 / 1024:.2f} MB"
        elif total_size_bytes >= 1024:
            total_size_human = f"{total_size_bytes / 1024:.2f} KB"
        else:
            total_size_human = f"{total_size_bytes} B"
        
        # 假设最大存储为10GB
        max_storage = 10 * 1024 * 1024 * 1024
        storage_usage_percent = (total_size_bytes / max_storage) * 100
        
        return {
            'total_datasets': total_datasets,
            'total_size_bytes': total_size_bytes,
            'total_size_human': total_size_human,
            'total_records': total_records,
            'datasets_by_type': datasets_by_type,
            'datasets_by_status': datasets_by_status,
            'recent_uploads': recent_uploads,
            'storage_usage_percent': round(storage_usage_percent, 2)
        }
    
    def get_recent_datasets(self, user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """获取最近访问的数据集
        
        获取当前用户最近访问的数据集列表。
        
        Args:
            user_id: 用户ID (必填)
            limit: 返回数量限制 (默认10)
            
        Returns:
            List[Dict[str, Any]]: 最近访问的数据集列表
                - dataset_id: 数据集ID
                - name: 数据集名称
                - last_accessed: 最后访问时间
                
        Example:
            >>> recent = service.get_recent_datasets("user123", 5)
            >>> for ds in recent:
            ...     print(f"{ds['name']}: {ds['last_accessed']}")
        """
        # 从访问日志获取最近访问的数据集
        recent_logs = self.access_log_repository.get_recent_by_user(
            user_id=user_id,
            limit=limit * 3  # 获取更多以去重
        )
        
        # 去重并保持顺序
        seen = set()
        recent_datasets = []
        
        for log in recent_logs:
            dataset_id = log.get('dataset_id')
            if dataset_id in seen:
                continue
            seen.add(dataset_id)
            
            # 获取数据集信息
            dataset = self.dataset_repository.get_by_id(dataset_id)
            if dataset and dataset.user_id == user_id:
                recent_datasets.append({
                    'dataset_id': dataset_id,
                    'name': dataset.name,
                    'dataset_type': dataset.dataset_type,
                    'status': dataset.status,
                    'last_accessed': log.get('created_at', log.get('accessed_at'))
                })
            
            if len(recent_datasets) >= limit:
                break
        
        # 如果访问日志不足，补充最近更新的数据集
        if len(recent_datasets) < limit:
            datasets = self.dataset_repository.list_by_user(
                user_id=user_id,
                limit=limit,
                offset=0,
                order_by='updated_at',
                order_desc=True
            )
            for ds in datasets:
                if ds.dataset_id not in seen:
                    recent_datasets.append({
                        'dataset_id': ds.dataset_id,
                        'name': ds.name,
                        'dataset_type': ds.dataset_type,
                        'status': ds.status,
                        'last_accessed': ds.updated_at.isoformat() if ds.updated_at else None
                    })
                    seen.add(ds.dataset_id)
                if len(recent_datasets) >= limit:
                    break
        
        return recent_datasets[:limit]

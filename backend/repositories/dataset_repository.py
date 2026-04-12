"""数据集仓库

实现数据集的持久化存储和检索操作，通过ORM方式操作数据库。
"""

import sys
import os
import logging
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime
import uuid

from sqlalchemy import or_, desc, asc
from sqlalchemy.sql import func
from sqlalchemy.exc import SQLAlchemyError

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from backend.modules.database.manager import get_database_manager
from backend.schemas.dataset import (
    DatasetEntity,
    DatasetVersionEntity,
    DatasetTagEntity,
    DatasetAccessLogEntity,
    Dataset,
    DatasetVersion,
    DatasetStatistics
)
from backend.modules.dataset.dataset_exceptions import DatasetNotFoundError, DatasetValidationError

logger = logging.getLogger(__name__)


class DatasetRepository:
    """数据集仓库
    
    提供数据集的CRUD操作，通过SQLAlchemy ORM与数据库交互。
    
    Attributes:
        _db_manager: 数据库管理器实例
        
    Example:
        >>> repo = DatasetRepository()
        >>> dataset = Dataset(name="test", user_id="user123")
        >>> created = repo.create(dataset)
        >>> print(created.dataset_id)
    """
    
    def __init__(self):
        """初始化数据集仓库
        
        获取数据库管理器实例，用于后续的数据库操作。
        """
        self._db_manager = get_database_manager()
        
    def create(self, dataset: Dataset, tenant_id: str = None) -> Dataset:
        """创建数据集
        
        将数据集DTO转换为实体并持久化到数据库。
        
        Args:
            dataset: 数据集DTO对象
                - user_id (str): 用户ID (必填)
                - name (str): 数据集名称 (必填)
                - description (str): 数据集描述 (可选)
                - dataset_type (str): 数据集类型，默认'text' (可选)
                - format (str): 数据格式，默认'json' (可选)
                - storage_path (str): 存储路径 (可选)
                - config (dict): 配置信息 (可选)
            tenant_id: 租户ID，默认使用user_id
            
        Returns:
            Dataset: 创建成功的数据集DTO对象，包含生成的dataset_id
            
        Raises:
            DatasetValidationError: 当必填字段缺失或验证失败时
            SQLAlchemyError: 当数据库操作失败时
            
        Example:
            >>> dataset = Dataset(name="training_data", user_id="user123")
            >>> created = repo.create(dataset)
            >>> print(created.dataset_id)  # 输出生成的UUID
        """
        with self._db_manager.get_db_session() as session:
            try:
                # 转换为实体
                entity = DatasetEntity.from_dto(dataset, tenant_id or dataset.user_id)
                
                # 如果没有ID，生成新的UUID
                if not entity.id:
                    entity.id = uuid.uuid4()
                
                session.add(entity)
                session.flush()  # 获取ID但不提交
                
                logger.info(f"Created dataset: {entity.id}, name: {entity.name}")
                return entity.to_dto()
                
            except SQLAlchemyError as e:
                logger.error(f"Failed to create dataset: {e}")
                raise
        
    def get_by_id(self, dataset_id: str) -> Optional[Dataset]:
        """根据ID获取数据集
        
        从数据库查询指定ID的数据集。
        
        Args:
            dataset_id: 数据集唯一标识符 (UUID字符串)
            
        Returns:
            Dataset: 数据集DTO对象，如果不存在则返回None
            
        Raises:
            SQLAlchemyError: 当数据库查询失败时
            
        Example:
            >>> dataset = repo.get_by_id("550e8400-e29b-41d4-a716-446655440000")
            >>> if dataset:
            ...     print(dataset.name)
        """
        with self._db_manager.get_db_session() as session:
            try:
                entity = session.query(DatasetEntity).filter(
                    DatasetEntity.id == dataset_id
                ).first()
                
                if entity:
                    return entity.to_dto()
                return None
                
            except SQLAlchemyError as e:
                logger.error(f"Failed to get dataset by id {dataset_id}: {e}")
                raise
        
    def list_by_user(
        self, 
        user_id: str, 
        limit: int = 50, 
        offset: int = 0,
        status: str = None,
        dataset_type: str = None,
        order_by: str = 'created_at',
        order_desc: bool = True
    ) -> List[Dataset]:
        """根据用户ID获取数据集列表
        
        支持分页、过滤和排序的数据集列表查询。
        
        Args:
            user_id: 用户ID
            limit: 返回数量限制，默认50，最大100
            offset: 偏移量，默认0
            status: 按状态过滤 (pending/uploading/processing/ready/error/archived)
            dataset_type: 按类型过滤 (text/image/audio/video/tabular/mixed)
            order_by: 排序字段，默认'created_at'，可选值:
                - 'created_at': 创建时间
                - 'updated_at': 更新时间
                - 'name': 名称
                - 'size': 大小
            order_desc: 是否降序排列，默认True
            
        Returns:
            List[Dataset]: 数据集DTO对象列表
            
        Raises:
            SQLAlchemyError: 当数据库查询失败时
            
        Example:
            >>> datasets = repo.list_by_user(
            ...     user_id="user123",
            ...     limit=10,
            ...     status="ready",
            ...     order_by="name"
            ... )
            >>> for ds in datasets:
            ...     print(ds.name)
        """
        with self._db_manager.get_db_session() as session:
            try:
                query = session.query(DatasetEntity).filter(
                    DatasetEntity.user_id == user_id
                )
                
                # 按状态过滤
                if status:
                    query = query.filter(DatasetEntity.status == status)
                    
                # 按类型过滤
                if dataset_type:
                    query = query.filter(DatasetEntity.dataset_type == dataset_type)
                
                # 排序
                order_column = getattr(DatasetEntity, order_by, DatasetEntity.created_at)
                if order_desc:
                    query = query.order_by(desc(order_column))
                else:
                    query = query.order_by(asc(order_column))
                
                # 分页
                entities = query.offset(offset).limit(limit).all()
                
                return [entity.to_dto() for entity in entities]
                
            except SQLAlchemyError as e:
                logger.error(f"Failed to list datasets for user {user_id}: {e}")
                raise
    
    def count_by_user(
        self, 
        user_id: str,
        status: str = None,
        dataset_type: str = None
    ) -> int:
        """统计用户数据集数量
        
        Args:
            user_id: 用户ID
            status: 按状态过滤 (可选)
            dataset_type: 按类型过滤 (可选)
            
        Returns:
            int: 数据集数量
            
        Example:
            >>> count = repo.count_by_user("user123", status="ready")
            >>> print(f"就绪的数据集数量: {count}")
        """
        with self._db_manager.get_db_session() as session:
            try:
                # pylint: disable=not-callable
                query = session.query(func.count(DatasetEntity.id)).filter(
                    DatasetEntity.user_id == user_id
                )
                
                if status:
                    query = query.filter(DatasetEntity.status == status)
                if dataset_type:
                    query = query.filter(DatasetEntity.dataset_type == dataset_type)
                    
                return query.scalar() or 0
                
            except SQLAlchemyError as e:
                logger.error(f"Failed to count datasets for user {user_id}: {e}")
                raise

    def count_by_tenant(self, tenant_id: str) -> int:
        """统计租户数据集数量
        
        Args:
            tenant_id: 租户ID
            
        Returns:
            int: 数据集数量
        """
        with self._db_manager.get_db_session() as session:
            try:
                # pylint: disable=not-callable
                query = session.query(func.count(DatasetEntity.id)).filter(
                    DatasetEntity.tenant_id == tenant_id
                )
                return query.scalar() or 0
            except SQLAlchemyError as e:
                logger.error(f"Failed to count datasets for tenant {tenant_id}: {e}")
                return 0
        
    def update(self, dataset: Dataset) -> Dataset:
        """更新数据集
        
        更新已存在的数据集信息。
        
        Args:
            dataset: 包含更新信息的数据集DTO对象
                - dataset_id (str): 数据集ID (必填)
                - 其他字段为可选，只更新非None的字段
            
        Returns:
            Dataset: 更新后的数据集DTO对象
            
        Raises:
            DatasetNotFoundError: 当数据集不存在时
            SQLAlchemyError: 当数据库操作失败时
            
        Example:
            >>> dataset = repo.get_by_id("550e8400-e29b-41d4-a716-446655440000")
            >>> dataset.name = "updated_name"
            >>> dataset.description = "新的描述"
            >>> updated = repo.update(dataset)
        """
        with self._db_manager.get_db_session() as session:
            try:
                entity = session.query(DatasetEntity).filter(
                    DatasetEntity.id == dataset.dataset_id
                ).first()
                
                if not entity:
                    raise DatasetNotFoundError(f"数据集 {dataset.dataset_id} 不存在")
                
                # 更新字段
                if dataset.name:
                    entity.name = dataset.name
                if dataset.description is not None:
                    entity.description = dataset.description
                if dataset.dataset_type:
                    entity.dataset_type = dataset.dataset_type
                if dataset.format:
                    entity.format = dataset.format
                if dataset.storage_path is not None:
                    entity.storage_path = dataset.storage_path
                if dataset.config is not None:
                    entity.config = dataset.config
                if dataset.status:
                    entity.status = dataset.status
                if dataset.ready is not None:
                    entity.ready = dataset.ready
                if dataset.size is not None:
                    entity.size = dataset.size
                if dataset.record_count is not None:
                    entity.record_count = dataset.record_count
                if dataset.features is not None:
                    entity.features = dataset.features
                if dataset.labels is not None:
                    entity.labels = dataset.labels
                if dataset.version is not None:
                    entity.version = dataset.version
                if dataset.checksum is not None:
                    entity.checksum = dataset.checksum
                if dataset.validated is not None:
                    entity.validated = dataset.validated
                if dataset.validation_result is not None:
                    entity.validation_result = dataset.validation_result
                
                # 更新时间戳会自动更新
                session.flush()
                
                logger.info(f"Updated dataset: {entity.id}")
                return entity.to_dto()
                
            except DatasetNotFoundError:
                raise
            except SQLAlchemyError as e:
                logger.error(f"Failed to update dataset {dataset.dataset_id}: {e}")
                raise
        
    def delete(self, dataset_id: str) -> bool:
        """删除数据集
        
        从数据库中删除指定的数据集。
        
        Args:
            dataset_id: 数据集ID
            
        Returns:
            bool: 删除成功返回True，数据集不存在返回False
            
        Raises:
            SQLAlchemyError: 当数据库操作失败时
            
        Example:
            >>> success = repo.delete("550e8400-e29b-41d4-a716-446655440000")
            >>> if success:
            ...     print("删除成功")
        """
        with self._db_manager.get_db_session() as session:
            try:
                entity = session.query(DatasetEntity).filter(
                    DatasetEntity.id == dataset_id
                ).first()
                
                if entity:
                    session.delete(entity)
                    logger.info(f"Deleted dataset: {dataset_id}")
                    return True
                return False
                
            except SQLAlchemyError as e:
                logger.error(f"Failed to delete dataset {dataset_id}: {e}")
                raise
        
    def exists(self, dataset_id: str) -> bool:
        """检查数据集是否存在
        
        Args:
            dataset_id: 数据集ID
            
        Returns:
            bool: 存在返回True，否则返回False
            
        Example:
            >>> if repo.exists("550e8400-e29b-41d4-a716-446655440000"):
            ...     print("数据集存在")
        """
        with self._db_manager.get_db_session() as session:
            try:
                # pylint: disable=not-callable
                count = session.query(func.count()).filter(
                    DatasetEntity.id == dataset_id
                ).scalar()
                return count > 0
                
            except SQLAlchemyError as e:
                logger.error(f"Failed to check dataset existence {dataset_id}: {e}")
                raise
    
    def search(
        self,
        user_id: str,
        keyword: str = None,
        status: str = None,
        dataset_type: str = None,
        start_date: datetime = None,
        end_date: datetime = None,
        limit: int = 50,
        offset: int = 0
    ) -> Tuple[List[Dataset], int]:
        """搜索数据集
        
        支持关键字搜索和多条件过滤的数据集查询。
        
        Args:
            user_id: 用户ID
            keyword: 搜索关键字，匹配名称和描述
            status: 按状态过滤
            dataset_type: 按类型过滤
            start_date: 创建时间起始
            end_date: 创建时间结束
            limit: 返回数量限制
            offset: 偏移量
            
        Returns:
            Tuple[List[Dataset], int]: (数据集列表, 总数)
            
        Example:
            >>> datasets, total = repo.search(
            ...     user_id="user123",
            ...     keyword="training",
            ...     status="ready"
            ... )
            >>> print(f"找到 {total} 个数据集")
        """
        with self._db_manager.get_db_session() as session:
            try:
                query = session.query(DatasetEntity).filter(
                    DatasetEntity.user_id == user_id
                )
                
                # 关键字搜索
                if keyword:
                    keyword_filter = or_(
                        DatasetEntity.name.ilike(f'%{keyword}%'),
                        DatasetEntity.description.ilike(f'%{keyword}%')
                    )
                    query = query.filter(keyword_filter)
                
                # 状态过滤
                if status:
                    query = query.filter(DatasetEntity.status == status)
                    
                # 类型过滤
                if dataset_type:
                    query = query.filter(DatasetEntity.dataset_type == dataset_type)
                    
                # 时间范围过滤
                if start_date:
                    query = query.filter(DatasetEntity.created_at >= start_date)
                if end_date:
                    query = query.filter(DatasetEntity.created_at <= end_date)
                
                # 获取总数
                total = query.count()
                
                # 分页和排序
                entities = query.order_by(desc(DatasetEntity.created_at))\
                    .offset(offset).limit(limit).all()
                
                return [entity.to_dto() for entity in entities], total
                
            except SQLAlchemyError as e:
                logger.error(f"Failed to search datasets: {e}")
                raise
    
    def update_status(self, dataset_id: str, status: str) -> bool:
        """更新数据集状态
        
        快捷方法，仅更新数据集的状态字段。
        
        Args:
            dataset_id: 数据集ID
            status: 新状态值 (pending/uploading/processing/ready/error/archived)
            
        Returns:
            bool: 更新成功返回True，数据集不存在返回False
            
        Example:
            >>> repo.update_status("550e8400-e29b-41d4-a716-446655440000", "ready")
        """
        with self._db_manager.get_db_session() as session:
            try:
                entity = session.query(DatasetEntity).filter(
                    DatasetEntity.id == dataset_id
                ).first()
                
                if entity:
                    entity.status = status
                    if status == 'ready':
                        entity.ready = True
                    elif status == 'error':
                        entity.ready = False
                    session.flush()
                    logger.info(f"Updated dataset {dataset_id} status to {status}")
                    return True
                return False
                
            except SQLAlchemyError as e:
                logger.error(f"Failed to update dataset status: {e}")
                raise
    
    def bulk_delete(self, dataset_ids: List[str]) -> int:
        """批量删除数据集
        
        Args:
            dataset_ids: 数据集ID列表
            
        Returns:
            int: 成功删除的数量
            
        Example:
            >>> ids = ["id1", "id2", "id3"]
            >>> deleted_count = repo.bulk_delete(ids)
            >>> print(f"删除了 {deleted_count} 个数据集")
        """
        with self._db_manager.get_db_session() as session:
            try:
                deleted = session.query(DatasetEntity).filter(
                    DatasetEntity.id.in_(dataset_ids)
                ).delete(synchronize_session='fetch')
                
                logger.info(f"Bulk deleted {deleted} datasets")
                return deleted
                
            except SQLAlchemyError as e:
                logger.error(f"Failed to bulk delete datasets: {e}")
                raise
    
    def get_statistics(self, user_id: str) -> Dict[str, Any]:
        """获取用户数据集统计信息
        
        Args:
            user_id: 用户ID
            
        Returns:
            Dict[str, Any]: 统计信息字典，包含:
                - total: 总数
                - by_status: 按状态分组的数量
                - by_type: 按类型分组的数量
                - total_size: 总大小(字节)
                - total_records: 总记录数
                
        Example:
            >>> stats = repo.get_statistics("user123")
            >>> print(f"总数据集: {stats['total']}")
            >>> print(f"就绪数据集: {stats['by_status'].get('ready', 0)}")
        """
        with self._db_manager.get_db_session() as session:
            try:
                # pylint: disable=not-callable
                # 总数
                total = session.query(func.count(DatasetEntity.id)).filter(
                    DatasetEntity.user_id == user_id
                ).scalar() or 0
                
                # 按状态分组
                status_counts = session.query(
                    DatasetEntity.status,
                    func.count(DatasetEntity.id).label('count')  # pylint: disable=not-callable
                ).filter(
                    DatasetEntity.user_id == user_id
                ).group_by(DatasetEntity.status).all()
                
                by_status = {status: count for status, count in status_counts}
                
                # 按类型分组
                type_counts = session.query(
                    DatasetEntity.dataset_type,
                    func.count(DatasetEntity.id)  # pylint: disable=not-callable
                ).filter(
                    DatasetEntity.user_id == user_id
                ).group_by(DatasetEntity.dataset_type).all()
                
                by_type = {dtype: count for dtype, count in type_counts}
                
                # 总大小和记录数
                size_records = session.query(
                    func.sum(DatasetEntity.size),  # pylint: disable=not-callable
                    func.sum(DatasetEntity.record_count)  # pylint: disable=not-callable
                ).filter(
                    DatasetEntity.user_id == user_id
                ).first()
                
                return {
                    'total': total,
                    'by_status': by_status,
                    'by_type': by_type,
                    'total_size': size_records[0] or 0,
                    'total_records': size_records[1] or 0
                }
                
            except SQLAlchemyError as e:
                logger.error(f"Failed to get statistics for user {user_id}: {e}")
                raise


class DatasetVersionRepository:
    """数据集版本仓库
    
    管理数据集版本历史的持久化操作。
    """
    
    def __init__(self):
        self._db_manager = get_database_manager()
    
    def create(self, version: DatasetVersion) -> DatasetVersion:
        """创建数据集版本
        
        Args:
            version: 版本DTO对象
            
        Returns:
            DatasetVersion: 创建的版本对象
        """
        with self._db_manager.get_db_session() as session:
            try:
                entity = DatasetVersionEntity(
                    dataset_id=uuid.UUID(version.dataset_id),
                    version=version.version,
                    description=version.description,
                    storage_path=version.storage_path,
                    size=version.size,
                    record_count=version.record_count,
                    checksum=version.checksum,
                    created_by=version.created_by,
                    changes=version.changes,
                    parent_version_id=uuid.UUID(version.parent_version_id) if version.parent_version_id else None
                )
                
                session.add(entity)
                session.flush()
                
                version.version_id = str(entity.id)
                version.created_at = entity.created_at
                
                logger.info(f"Created dataset version: {entity.id}")
                return version
                
            except SQLAlchemyError as e:
                logger.error(f"Failed to create dataset version: {e}")
                raise
    
    def get_by_id(self, version_id: str) -> Optional[DatasetVersion]:
        """根据ID获取版本"""
        with self._db_manager.get_db_session() as session:
            try:
                entity = session.query(DatasetVersionEntity).filter(
                    DatasetVersionEntity.id == version_id
                ).first()
                
                if entity:
                    return DatasetVersion(
                        version_id=str(entity.id),
                        dataset_id=str(entity.dataset_id),
                        version=entity.version,
                        description=entity.description,
                        storage_path=entity.storage_path,
                        size=entity.size,
                        record_count=entity.record_count,
                        checksum=entity.checksum,
                        created_by=entity.created_by,
                        created_at=entity.created_at,
                        changes=entity.changes,
                        parent_version_id=str(entity.parent_version_id) if entity.parent_version_id else None
                    )
                return None
                
            except SQLAlchemyError as e:
                logger.error(f"Failed to get version {version_id}: {e}")
                raise
    
    def list_by_dataset(self, dataset_id: str) -> List[DatasetVersion]:
        """获取数据集的所有版本"""
        with self._db_manager.get_db_session() as session:
            try:
                entities = session.query(DatasetVersionEntity).filter(
                    DatasetVersionEntity.dataset_id == dataset_id
                ).order_by(desc(DatasetVersionEntity.created_at)).all()
                
                return [
                    DatasetVersion(
                        version_id=str(e.id),
                        dataset_id=str(e.dataset_id),
                        version=e.version,
                        description=e.description,
                        storage_path=e.storage_path,
                        size=e.size,
                        record_count=e.record_count,
                        checksum=e.checksum,
                        created_by=e.created_by,
                        created_at=e.created_at,
                        changes=e.changes,
                        parent_version_id=str(e.parent_version_id) if e.parent_version_id else None
                    )
                    for e in entities
                ]
                
            except SQLAlchemyError as e:
                logger.error(f"Failed to list versions for dataset {dataset_id}: {e}")
                raise


class DatasetAccessLogRepository:
    """数据集访问日志仓库
    
    记录数据集的访问历史。
    """
    
    def __init__(self):
        self._db_manager = get_database_manager()
    
    def create(
        self, 
        dataset_id: str, 
        user_id: str, 
        action: str,
        details: Dict[str, Any] = None,
        ip_address: str = None,
        user_agent: str = None
    ) -> str:
        """创建访问日志
        
        Args:
            dataset_id: 数据集ID
            user_id: 用户ID
            action: 操作类型 (read/write/delete/download/process)
            details: 操作详情
            ip_address: IP地址
            user_agent: 用户代理
            
        Returns:
            str: 日志ID
        """
        with self._db_manager.get_db_session() as session:
            try:
                entity = DatasetAccessLogEntity(
                    dataset_id=uuid.UUID(dataset_id),
                    user_id=user_id,
                    action=action,
                    details=details,
                    ip_address=ip_address,
                    user_agent=user_agent
                )
                
                session.add(entity)
                session.flush()
                
                return str(entity.id)
                
            except SQLAlchemyError as e:
                logger.error(f"Failed to create access log: {e}")
                raise
    
    def list_by_dataset(
        self, 
        dataset_id: str, 
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """获取数据集访问日志列表"""
        with self._db_manager.get_db_session() as session:
            try:
                entities = session.query(DatasetAccessLogEntity).filter(
                    DatasetAccessLogEntity.dataset_id == dataset_id
                ).order_by(
                    desc(DatasetAccessLogEntity.timestamp)
                ).offset(offset).limit(limit).all()
                
                return [
                    {
                        'log_id': str(e.id),
                        'dataset_id': str(e.dataset_id),
                        'user_id': e.user_id,
                        'action': e.action,
                        'details': e.details,
                        'ip_address': e.ip_address,
                        'timestamp': e.timestamp.isoformat() if e.timestamp else None
                    }
                    for e in entities
                ]
                
            except SQLAlchemyError as e:
                logger.error(f"Failed to list access logs: {e}")
                raise
    
    def get_recent_by_user(
        self, 
        user_id: str, 
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """获取用户最近的访问日志
        
        Args:
            user_id: 用户ID
            limit: 返回数量限制
            
        Returns:
            List[Dict[str, Any]]: 访问日志列表
        """
        with self._db_manager.get_db_session() as session:
            try:
                entities = session.query(DatasetAccessLogEntity).filter(
                    DatasetAccessLogEntity.user_id == user_id
                ).order_by(
                    desc(DatasetAccessLogEntity.timestamp)
                ).limit(limit).all()
                
                return [
                    {
                        'log_id': str(e.id),
                        'dataset_id': str(e.dataset_id),
                        'user_id': e.user_id,
                        'action': e.action,
                        'details': e.details,
                        'ip_address': e.ip_address,
                        'timestamp': e.timestamp.isoformat() if e.timestamp else None
                    }
                    for e in entities
                ]
                
            except SQLAlchemyError as e:
                logger.error(f"Failed to get recent logs by user: {e}")
                raise


class DatasetTagRepository:
    """数据集标签仓库
    
    管理数据集标签的持久化操作。
    """
    
    def __init__(self):
        self._db_manager = get_database_manager()
    
    def add_tag(
        self, 
        dataset_id: str, 
        tag_name: str, 
        tag_value: str = None,
        created_by: str = None
    ) -> str:
        """添加标签
        
        Args:
            dataset_id: 数据集ID
            tag_name: 标签名称
            tag_value: 标签值
            created_by: 创建者ID
            
        Returns:
            str: 标签ID
        """
        with self._db_manager.get_db_session() as session:
            try:
                entity = DatasetTagEntity(
                    dataset_id=uuid.UUID(dataset_id),
                    tag_name=tag_name,
                    tag_value=tag_value,
                    created_by=created_by
                )
                
                session.add(entity)
                session.flush()
                
                return str(entity.id)
                
            except SQLAlchemyError as e:
                logger.error(f"Failed to add tag: {e}")
                raise
    
    def remove_tag(self, dataset_id: str, tag_name: str) -> bool:
        """移除标签"""
        with self._db_manager.get_db_session() as session:
            try:
                deleted = session.query(DatasetTagEntity).filter(
                    DatasetTagEntity.dataset_id == dataset_id,
                    DatasetTagEntity.tag_name == tag_name
                ).delete()
                
                return deleted > 0
                
            except SQLAlchemyError as e:
                logger.error(f"Failed to remove tag: {e}")
                raise
    
    def get_tags(self, dataset_id: str) -> List[Dict[str, str]]:
        """获取数据集的所有标签"""
        with self._db_manager.get_db_session() as session:
            try:
                entities = session.query(DatasetTagEntity).filter(
                    DatasetTagEntity.dataset_id == dataset_id
                ).all()
                
                return [
                    {
                        'tag_id': str(e.id),
                        'tag_name': e.tag_name,
                        'tag_value': e.tag_value
                    }
                    for e in entities
                ]
                
            except SQLAlchemyError as e:
                logger.error(f"Failed to get tags: {e}")
                raise
    
    def find_by_tag(
        self, 
        user_id: str, 
        tag_name: str, 
        tag_value: str = None
    ) -> List[str]:
        """根据标签查找数据集ID列表"""
        with self._db_manager.get_db_session() as session:
            try:
                query = session.query(DatasetTagEntity.dataset_id).join(
                    DatasetEntity,
                    DatasetTagEntity.dataset_id == DatasetEntity.id
                ).filter(
                    DatasetEntity.user_id == user_id,
                    DatasetTagEntity.tag_name == tag_name
                )
                
                if tag_value:
                    query = query.filter(DatasetTagEntity.tag_value == tag_value)
                
                results = query.distinct().all()
                
                return [str(r[0]) for r in results]
                
            except SQLAlchemyError as e:
                logger.error(f"Failed to find by tag: {e}")
                raise

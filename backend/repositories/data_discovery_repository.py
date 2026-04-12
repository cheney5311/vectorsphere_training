"""数据发现仓库层

实现数据发现记录、数据源配置、同步任务等的持久化操作。
支持内存存储和数据库持久化两种模式。
"""

import sys
import os
import uuid
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from contextlib import contextmanager

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from backend.modules.dataset.dataset_exceptions import (
    DiscoveryNotFoundError,
    DataSourceNotFoundError,
)

logger = logging.getLogger(__name__)


# ============================================================================
# 数据源仓库
# ============================================================================

class DataSourceRepository:
    """数据源仓库
    
    管理数据源配置的CRUD操作。
    支持内存存储和数据库持久化两种模式。
    """
    
    def __init__(self, db_service=None, use_memory_storage: bool = False):
        """初始化数据源仓库
        
        Args:
            db_service: 数据库服务实例
            use_memory_storage: 是否使用内存存储
        """
        self._use_memory_storage = use_memory_storage
        self._db_service = None
        self._sources: Dict[str, Dict[str, Any]] = {}
        
        if not use_memory_storage:
            self._init_database(db_service)
    
    def _init_database(self, db_service=None):
        """初始化数据库连接"""
        try:
            if db_service:
                self._db_service = db_service
            else:
                from backend.modules.database.manager import get_database_manager
                self._db_service = get_database_manager()
        except ImportError as e:
            logger.warning(f"Database manager not available: {e}, using memory storage")
            self._use_memory_storage = True
    
    @contextmanager
    def _get_session(self):
        """获取数据库会话"""
        if self._db_service:
            with self._db_service.get_db_session() as session:
                yield session
        else:
            yield None
    
    def create(self, source_data: Dict[str, Any]) -> Dict[str, Any]:
        """创建数据源
        
        Args:
            source_data: 数据源数据字典
            
        Returns:
            创建的数据源数据
        """
        if not source_data.get('source_id'):
            source_data['source_id'] = str(uuid.uuid4())
        source_data['created_at'] = datetime.utcnow()
        
        if self._use_memory_storage:
            self._sources[source_data['source_id']] = source_data.copy()
            logger.info(f"Created data source (memory): {source_data['source_id']}")
            return source_data
        
        try:
            from backend.schemas.data_discovery_db_models import DataSource
            
            with self._get_session() as session:
                if session:
                    db_source = DataSource(
                        id=uuid.UUID(source_data['source_id']),
                        user_id=source_data.get('user_id', ''),
                        tenant_id=source_data.get('tenant_id'),
                        source_type=source_data.get('source_type', 'file_system'),
                        location=source_data.get('location', ''),
                        name=source_data.get('name', ''),
                        description=source_data.get('description'),
                        credentials=source_data.get('credentials'),
                        config=source_data.get('config', {}),
                        status=source_data.get('status', 'active'),
                        last_scan_at=source_data.get('last_scan_at'),
                        last_scan_result=source_data.get('last_scan_result'),
                    )
                    session.add(db_source)
                    session.flush()
                    logger.info(f"Created data source (db): {source_data['source_id']}")
                    return db_source.to_dict()
        except Exception as e:
            logger.error(f"Failed to create data source in database: {e}")
            # 回退到内存存储
            self._sources[source_data['source_id']] = source_data.copy()
        
        return source_data
    
    def get_by_id(self, source_id: str) -> Optional[Dict[str, Any]]:
        """根据ID获取数据源
        
        Args:
            source_id: 数据源ID
            
        Returns:
            数据源数据，不存在则返回None
        """
        if self._use_memory_storage:
            return self._sources.get(source_id)
        
        try:
            from backend.schemas.data_discovery_db_models import DataSource
            
            with self._get_session() as session:
                if session:
                    db_source = session.query(DataSource).filter(
                        DataSource.id == uuid.UUID(source_id)
                    ).first()
                    if db_source:
                        return db_source.to_dict()
        except Exception as e:
            logger.error(f"Failed to get data source from database: {e}")
            return self._sources.get(source_id)
        
        return None
    
    def get_by_user(
        self, 
        user_id: str, 
        tenant_id: Optional[str] = None,
        limit: int = 50,
        offset: int = 0
    ) -> Tuple[List[Dict[str, Any]], int]:
        """获取用户的数据源列表
        
        Args:
            user_id: 用户ID
            tenant_id: 租户ID（可选）
            limit: 返回数量限制
            offset: 偏移量
            
        Returns:
            数据源列表和总数
        """
        if self._use_memory_storage:
            sources = [
                s for s in self._sources.values()
                if s.get('user_id') == user_id and (tenant_id is None or s.get('tenant_id') == tenant_id)
            ]
            sources.sort(key=lambda x: x.get('created_at') or datetime.min, reverse=True)
            total = len(sources)
            return sources[offset:offset + limit], total
        
        try:
            from backend.schemas.data_discovery_db_models import DataSource
            from sqlalchemy import desc
            
            with self._get_session() as session:
                if session:
                    query = session.query(DataSource).filter(DataSource.user_id == user_id)
                    if tenant_id:
                        query = query.filter(DataSource.tenant_id == tenant_id)
                    
                    total = query.count()
                    sources = query.order_by(desc(DataSource.created_at)).offset(offset).limit(limit).all()
                    return [s.to_dict() for s in sources], total
        except Exception as e:
            logger.error(f"Failed to get data sources from database: {e}")
            # 回退到内存
            sources = [s for s in self._sources.values() if s.get('user_id') == user_id]
            return sources[offset:offset + limit], len(sources)
        
        return [], 0
    
    def update(self, source_id: str, update_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """更新数据源
        
        Args:
            source_id: 数据源ID
            update_data: 更新数据
            
        Returns:
            更新后的数据源数据
        """
        if self._use_memory_storage:
            if source_id not in self._sources:
                raise DataSourceNotFoundError(source_id)
            self._sources[source_id].update(update_data)
            self._sources[source_id]['updated_at'] = datetime.utcnow()
            logger.info(f"Updated data source (memory): {source_id}")
            return self._sources[source_id]
        
        try:
            from backend.schemas.data_discovery_db_models import DataSource
            
            with self._get_session() as session:
                if session:
                    db_source = session.query(DataSource).filter(
                        DataSource.id == uuid.UUID(source_id)
                    ).first()
                    if not db_source:
                        raise DataSourceNotFoundError(source_id)
                    
                    for key, value in update_data.items():
                        if hasattr(db_source, key):
                            setattr(db_source, key, value)
                    
                    session.flush()
                    logger.info(f"Updated data source (db): {source_id}")
                    return db_source.to_dict()
        except DataSourceNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Failed to update data source in database: {e}")
            if source_id in self._sources:
                self._sources[source_id].update(update_data)
                return self._sources[source_id]
        
        return None
    
    def delete(self, source_id: str) -> bool:
        """删除数据源
        
        Args:
            source_id: 数据源ID
            
        Returns:
            删除成功返回True
        """
        if self._use_memory_storage:
            if source_id in self._sources:
                del self._sources[source_id]
                logger.info(f"Deleted data source (memory): {source_id}")
                return True
            return False
        
        try:
            from backend.schemas.data_discovery_db_models import DataSource
            
            with self._get_session() as session:
                if session:
                    db_source = session.query(DataSource).filter(
                        DataSource.id == uuid.UUID(source_id)
                    ).first()
                    if db_source:
                        session.delete(db_source)
                        logger.info(f"Deleted data source (db): {source_id}")
                        return True
        except Exception as e:
            logger.error(f"Failed to delete data source from database: {e}")
            if source_id in self._sources:
                del self._sources[source_id]
                return True
        
        return False
    
    def update_scan_result(
        self, 
        source_id: str, 
        scan_result: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """更新数据源的扫描结果
        
        Args:
            source_id: 数据源ID
            scan_result: 扫描结果
            
        Returns:
            更新后的数据源数据
        """
        return self.update(source_id, {
            'last_scan_at': datetime.utcnow(),
            'last_scan_result': scan_result
        })


# ============================================================================
# 发现记录仓库
# ============================================================================

class DiscoveryRecordRepository:
    """发现记录仓库
    
    管理数据发现记录的CRUD操作。
    """
    
    def __init__(self, db_service=None, use_memory_storage: bool = False):
        """初始化发现记录仓库"""
        self._use_memory_storage = use_memory_storage
        self._db_service = None
        self._records: Dict[str, Dict[str, Any]] = {}
        
        if not use_memory_storage:
            self._init_database(db_service)
    
    def _init_database(self, db_service=None):
        """初始化数据库连接"""
        try:
            if db_service:
                self._db_service = db_service
            else:
                from backend.modules.database.manager import get_database_manager
                self._db_service = get_database_manager()
        except ImportError as e:
            logger.warning(f"Database manager not available: {e}, using memory storage")
            self._use_memory_storage = True
    
    @contextmanager
    def _get_session(self):
        """获取数据库会话"""
        if self._db_service:
            with self._db_service.get_db_session() as session:
                yield session
        else:
            yield None
    
    def create(self, record_data: Dict[str, Any]) -> Dict[str, Any]:
        """创建发现记录
        
        Args:
            record_data: 记录数据字典
            
        Returns:
            创建的记录数据
        """
        if not record_data.get('record_id'):
            record_data['record_id'] = str(uuid.uuid4())
        record_data['created_at'] = datetime.utcnow()
        
        if self._use_memory_storage:
            self._records[record_data['record_id']] = record_data.copy()
            logger.info(f"Created discovery record (memory): {record_data['record_id']}")
            return record_data
        
        try:
            from backend.schemas.data_discovery_db_models import DiscoveryRecord
            
            with self._get_session() as session:
                if session:
                    db_record = DiscoveryRecord(
                        id=uuid.UUID(record_data['record_id']),
                        user_id=record_data.get('user_id', ''),
                        tenant_id=record_data.get('tenant_id'),
                        source_id=uuid.UUID(record_data['source_id']) if record_data.get('source_id') else None,
                        source_type=record_data.get('source_type', 'file_system'),
                        source_location=record_data.get('source_location', ''),
                        status=record_data.get('status', 'pending'),
                        datasets_discovered=record_data.get('datasets_discovered', 0),
                        datasets_ingested=record_data.get('datasets_ingested', 0),
                        discovered_items=record_data.get('discovered_items', []),
                        scan_config=record_data.get('scan_config', {}),
                        error_message=record_data.get('error_message'),
                        metadata_=record_data.get('metadata', {}),
                    )
                    session.add(db_record)
                    session.flush()
                    logger.info(f"Created discovery record (db): {record_data['record_id']}")
                    return db_record.to_dict()
        except Exception as e:
            logger.error(f"Failed to create discovery record in database: {e}")
            self._records[record_data['record_id']] = record_data.copy()
        
        return record_data
    
    def get_by_id(self, record_id: str) -> Optional[Dict[str, Any]]:
        """根据ID获取发现记录
        
        Args:
            record_id: 记录ID
            
        Returns:
            发现记录数据，不存在则返回None
        """
        if self._use_memory_storage:
            return self._records.get(record_id)
        
        try:
            from backend.schemas.data_discovery_db_models import DiscoveryRecord
            
            with self._get_session() as session:
                if session:
                    db_record = session.query(DiscoveryRecord).filter(
                        DiscoveryRecord.id == uuid.UUID(record_id)
                    ).first()
                    if db_record:
                        return db_record.to_dict()
        except Exception as e:
            logger.error(f"Failed to get discovery record from database: {e}")
            return self._records.get(record_id)
        
        return None
    
    def get_by_user(
        self,
        user_id: str,
        tenant_id: Optional[str] = None,
        status_filter: Optional[List[str]] = None,
        source_type_filter: Optional[List[str]] = None,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
        limit: int = 50,
        offset: int = 0
    ) -> Tuple[List[Dict[str, Any]], int]:
        """获取用户的发现记录列表
        
        Args:
            user_id: 用户ID
            tenant_id: 租户ID（可选）
            status_filter: 状态过滤列表
            source_type_filter: 数据源类型过滤列表
            date_from: 开始日期
            date_to: 结束日期
            limit: 返回数量限制
            offset: 偏移量
            
        Returns:
            发现记录列表和总数
        """
        if self._use_memory_storage:
            filtered = []
            for record in self._records.values():
                if record.get('user_id') != user_id:
                    continue
                if tenant_id is not None and record.get('tenant_id') != tenant_id:
                    continue
                if status_filter and record.get('status') not in status_filter:
                    continue
                if source_type_filter and record.get('source_type') not in source_type_filter:
                    continue
                created_at = record.get('created_at')
                if date_from and created_at and created_at < date_from:
                    continue
                if date_to and created_at and created_at > date_to:
                    continue
                filtered.append(record)
            
            filtered.sort(key=lambda x: x.get('created_at') or datetime.min, reverse=True)
            total = len(filtered)
            return filtered[offset:offset + limit], total
        
        try:
            from backend.schemas.data_discovery_db_models import DiscoveryRecord
            from sqlalchemy import desc
            
            with self._get_session() as session:
                if session:
                    query = session.query(DiscoveryRecord).filter(
                        DiscoveryRecord.user_id == user_id
                    )
                    if tenant_id:
                        query = query.filter(DiscoveryRecord.tenant_id == tenant_id)
                    if status_filter:
                        query = query.filter(DiscoveryRecord.status.in_(status_filter))
                    if source_type_filter:
                        query = query.filter(DiscoveryRecord.source_type.in_(source_type_filter))
                    if date_from:
                        query = query.filter(DiscoveryRecord.created_at >= date_from)
                    if date_to:
                        query = query.filter(DiscoveryRecord.created_at <= date_to)
                    
                    total = query.count()
                    records = query.order_by(desc(DiscoveryRecord.created_at)).offset(offset).limit(limit).all()
                    return [r.to_dict() for r in records], total
        except Exception as e:
            logger.error(f"Failed to get discovery records from database: {e}")
        
        return [], 0
    
    def update(self, record_id: str, update_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """更新发现记录
        
        Args:
            record_id: 记录ID
            update_data: 更新数据
            
        Returns:
            更新后的记录数据
        """
        if self._use_memory_storage:
            if record_id not in self._records:
                raise DiscoveryNotFoundError(record_id)
            self._records[record_id].update(update_data)
            self._records[record_id]['updated_at'] = datetime.utcnow()
            logger.info(f"Updated discovery record (memory): {record_id}")
            return self._records[record_id]
        
        try:
            from backend.schemas.data_discovery_db_models import DiscoveryRecord
            
            with self._get_session() as session:
                if session:
                    db_record = session.query(DiscoveryRecord).filter(
                        DiscoveryRecord.id == uuid.UUID(record_id)
                    ).first()
                    if not db_record:
                        raise DiscoveryNotFoundError(record_id)
                    
                    for key, value in update_data.items():
                        if key == 'metadata':
                            setattr(db_record, 'metadata_', value)
                        elif hasattr(db_record, key):
                            setattr(db_record, key, value)
                    
                    session.flush()
                    logger.info(f"Updated discovery record (db): {record_id}")
                    return db_record.to_dict()
        except DiscoveryNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Failed to update discovery record in database: {e}")
        
        return None
    
    def update_status(
        self,
        record_id: str,
        status: str,
        error_message: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """更新发现记录状态
        
        Args:
            record_id: 记录ID
            status: 新状态
            error_message: 错误信息（可选）
            
        Returns:
            更新后的记录数据
        """
        update_data = {'status': status, 'updated_at': datetime.utcnow()}
        if error_message:
            update_data['error_message'] = error_message
        if status in ('discovered', 'failed', 'completed'):
            update_data['completed_at'] = datetime.utcnow()
        
        return self.update(record_id, update_data)
    
    def add_discovered_items(
        self,
        record_id: str,
        items: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """添加发现的数据项
        
        Args:
            record_id: 记录ID
            items: 发现的数据项列表
            
        Returns:
            更新后的记录数据
        """
        record = self.get_by_id(record_id)
        if not record:
            return None
        
        discovered_items = record.get('discovered_items', [])
        discovered_items.extend(items)
        
        return self.update(record_id, {
            'discovered_items': discovered_items,
            'datasets_discovered': len(discovered_items)
        })
    
    def delete(self, record_id: str) -> bool:
        """删除发现记录
        
        Args:
            record_id: 记录ID
            
        Returns:
            删除成功返回True
        """
        if self._use_memory_storage:
            if record_id in self._records:
                del self._records[record_id]
                logger.info(f"Deleted discovery record (memory): {record_id}")
                return True
            return False
        
        try:
            from backend.schemas.data_discovery_db_models import DiscoveryRecord
            
            with self._get_session() as session:
                if session:
                    db_record = session.query(DiscoveryRecord).filter(
                        DiscoveryRecord.id == uuid.UUID(record_id)
                    ).first()
                    if db_record:
                        session.delete(db_record)
                        logger.info(f"Deleted discovery record (db): {record_id}")
                        return True
        except Exception as e:
            logger.error(f"Failed to delete discovery record from database: {e}")
        
        return False


# ============================================================================
# 发现的数据集仓库
# ============================================================================

class DiscoveredDatasetRepository:
    """发现的数据集仓库
    
    管理发现的数据集实体的CRUD操作。
    """
    
    def __init__(self, db_service=None, use_memory_storage: bool = False):
        """初始化发现的数据集仓库"""
        self._use_memory_storage = use_memory_storage
        self._db_service = None
        self._datasets: Dict[str, Dict[str, Any]] = {}
        
        if not use_memory_storage:
            self._init_database(db_service)
    
    def _init_database(self, db_service=None):
        """初始化数据库连接"""
        try:
            if db_service:
                self._db_service = db_service
            else:
                from backend.modules.database.manager import get_database_manager
                self._db_service = get_database_manager()
        except ImportError as e:
            logger.warning(f"Database manager not available: {e}, using memory storage")
            self._use_memory_storage = True
    
    @contextmanager
    def _get_session(self):
        """获取数据库会话"""
        if self._db_service:
            with self._db_service.get_db_session() as session:
                yield session
        else:
            yield None
    
    def create(self, dataset_data: Dict[str, Any]) -> Dict[str, Any]:
        """创建发现的数据集记录
        
        Args:
            dataset_data: 数据集数据字典
            
        Returns:
            创建的数据集数据
        """
        if not dataset_data.get('discovery_id'):
            dataset_data['discovery_id'] = str(uuid.uuid4())
        dataset_data['discovered_at'] = datetime.utcnow()
        
        if self._use_memory_storage:
            self._datasets[dataset_data['discovery_id']] = dataset_data.copy()
            logger.info(f"Created discovered dataset (memory): {dataset_data['discovery_id']}")
            return dataset_data
        
        try:
            from backend.schemas.data_discovery_db_models import DiscoveredDataset
            
            with self._get_session() as session:
                if session:
                    db_dataset = DiscoveredDataset(
                        id=uuid.UUID(dataset_data['discovery_id']),
                        record_id=uuid.UUID(dataset_data['record_id']) if dataset_data.get('record_id') else None,
                        user_id=dataset_data.get('user_id', ''),
                        tenant_id=dataset_data.get('tenant_id'),
                        dataset_name=dataset_data.get('dataset_name', ''),
                        source_id=dataset_data.get('source_id', ''),
                        source_type=dataset_data.get('source_type', 'file_system'),
                        source_path=dataset_data.get('source_path', ''),
                        data_format=dataset_data.get('data_format', 'unknown'),
                        size_bytes=dataset_data.get('size_bytes', 0),
                        row_count=dataset_data.get('row_count'),
                        column_count=dataset_data.get('column_count'),
                        schema_info=dataset_data.get('schema_info'),
                        preview_data=dataset_data.get('preview_data'),
                        quality_score=dataset_data.get('quality_score'),
                        completeness=dataset_data.get('completeness'),
                        status=dataset_data.get('status', 'discovered'),
                        ingested_dataset_id=dataset_data.get('ingested_dataset_id'),
                        metadata_=dataset_data.get('metadata', {}),
                    )
                    session.add(db_dataset)
                    session.flush()
                    logger.info(f"Created discovered dataset (db): {dataset_data['discovery_id']}")
                    return db_dataset.to_dict()
        except Exception as e:
            logger.error(f"Failed to create discovered dataset in database: {e}")
            self._datasets[dataset_data['discovery_id']] = dataset_data.copy()
        
        return dataset_data
    
    def create_batch(self, datasets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """批量创建发现的数据集记录
        
        Args:
            datasets: 数据集数据列表
            
        Returns:
            创建的数据集数据列表
        """
        created = []
        for dataset in datasets:
            created.append(self.create(dataset))
        return created
    
    def get_by_id(self, discovery_id: str) -> Optional[Dict[str, Any]]:
        """根据ID获取发现的数据集
        
        Args:
            discovery_id: 发现ID
            
        Returns:
            发现的数据集数据，不存在则返回None
        """
        if self._use_memory_storage:
            return self._datasets.get(discovery_id)
        
        try:
            from backend.schemas.data_discovery_db_models import DiscoveredDataset
            
            with self._get_session() as session:
                if session:
                    db_dataset = session.query(DiscoveredDataset).filter(
                        DiscoveredDataset.id == uuid.UUID(discovery_id)
                    ).first()
                    if db_dataset:
                        return db_dataset.to_dict()
        except Exception as e:
            logger.error(f"Failed to get discovered dataset from database: {e}")
            return self._datasets.get(discovery_id)
        
        return None
    
    def get_by_record(self, record_id: str) -> List[Dict[str, Any]]:
        """获取发现记录下的所有数据集
        
        Args:
            record_id: 发现记录ID
            
        Returns:
            发现的数据集列表
        """
        if self._use_memory_storage:
            return [d for d in self._datasets.values() if d.get('record_id') == record_id]
        
        try:
            from backend.schemas.data_discovery_db_models import DiscoveredDataset
            
            with self._get_session() as session:
                if session:
                    datasets = session.query(DiscoveredDataset).filter(
                        DiscoveredDataset.record_id == uuid.UUID(record_id)
                    ).all()
                    return [d.to_dict() for d in datasets]
        except Exception as e:
            logger.error(f"Failed to get discovered datasets from database: {e}")
        
        return []
    
    def get_by_user(
        self,
        user_id: str,
        tenant_id: Optional[str] = None,
        status_filter: Optional[List[str]] = None,
        format_filter: Optional[List[str]] = None,
        limit: int = 50,
        offset: int = 0
    ) -> Tuple[List[Dict[str, Any]], int]:
        """获取用户发现的数据集列表
        
        Args:
            user_id: 用户ID
            tenant_id: 租户ID（可选）
            status_filter: 状态过滤
            format_filter: 格式过滤
            limit: 返回数量限制
            offset: 偏移量
            
        Returns:
            发现的数据集列表和总数
        """
        if self._use_memory_storage:
            filtered = []
            for dataset in self._datasets.values():
                if dataset.get('user_id') != user_id:
                    continue
                if tenant_id is not None and dataset.get('tenant_id') != tenant_id:
                    continue
                if status_filter and dataset.get('status') not in status_filter:
                    continue
                if format_filter and dataset.get('data_format') not in format_filter:
                    continue
                filtered.append(dataset)
            
            filtered.sort(key=lambda x: x.get('discovered_at') or datetime.min, reverse=True)
            total = len(filtered)
            return filtered[offset:offset + limit], total
        
        try:
            from backend.schemas.data_discovery_db_models import DiscoveredDataset
            from sqlalchemy import desc
            
            with self._get_session() as session:
                if session:
                    query = session.query(DiscoveredDataset).filter(
                        DiscoveredDataset.user_id == user_id
                    )
                    if tenant_id:
                        query = query.filter(DiscoveredDataset.tenant_id == tenant_id)
                    if status_filter:
                        query = query.filter(DiscoveredDataset.status.in_(status_filter))
                    if format_filter:
                        query = query.filter(DiscoveredDataset.data_format.in_(format_filter))
                    
                    total = query.count()
                    datasets = query.order_by(desc(DiscoveredDataset.discovered_at)).offset(offset).limit(limit).all()
                    return [d.to_dict() for d in datasets], total
        except Exception as e:
            logger.error(f"Failed to get discovered datasets from database: {e}")
        
        return [], 0
    
    def update(self, discovery_id: str, update_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """更新发现的数据集
        
        Args:
            discovery_id: 发现ID
            update_data: 更新数据
            
        Returns:
            更新后的数据集数据
        """
        if self._use_memory_storage:
            if discovery_id in self._datasets:
                self._datasets[discovery_id].update(update_data)
                logger.info(f"Updated discovered dataset (memory): {discovery_id}")
                return self._datasets[discovery_id]
            return None
        
        try:
            from backend.schemas.data_discovery_db_models import DiscoveredDataset
            
            with self._get_session() as session:
                if session:
                    db_dataset = session.query(DiscoveredDataset).filter(
                        DiscoveredDataset.id == uuid.UUID(discovery_id)
                    ).first()
                    if not db_dataset:
                        return None
                    
                    for key, value in update_data.items():
                        if key == 'metadata':
                            setattr(db_dataset, 'metadata_', value)
                        elif hasattr(db_dataset, key):
                            setattr(db_dataset, key, value)
                    
                    session.flush()
                    logger.info(f"Updated discovered dataset (db): {discovery_id}")
                    return db_dataset.to_dict()
        except Exception as e:
            logger.error(f"Failed to update discovered dataset in database: {e}")
        
        return None
    
    def update_status(
        self,
        discovery_id: str,
        status: str,
        ingested_dataset_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """更新发现的数据集状态
        
        Args:
            discovery_id: 发现ID
            status: 新状态
            ingested_dataset_id: 接入后的数据集ID（可选）
            
        Returns:
            更新后的数据集数据
        """
        update_data = {'status': status}
        if ingested_dataset_id:
            update_data['ingested_dataset_id'] = ingested_dataset_id
        
        return self.update(discovery_id, update_data)
    
    def delete(self, discovery_id: str) -> bool:
        """删除发现的数据集记录
        
        Args:
            discovery_id: 发现ID
            
        Returns:
            删除成功返回True
        """
        if self._use_memory_storage:
            if discovery_id in self._datasets:
                del self._datasets[discovery_id]
                logger.info(f"Deleted discovered dataset (memory): {discovery_id}")
                return True
            return False
        
        try:
            from backend.schemas.data_discovery_db_models import DiscoveredDataset
            
            with self._get_session() as session:
                if session:
                    db_dataset = session.query(DiscoveredDataset).filter(
                        DiscoveredDataset.id == uuid.UUID(discovery_id)
                    ).first()
                    if db_dataset:
                        session.delete(db_dataset)
                        logger.info(f"Deleted discovered dataset (db): {discovery_id}")
                        return True
        except Exception as e:
            logger.error(f"Failed to delete discovered dataset from database: {e}")
        
        return False


# ============================================================================
# 同步配置仓库
# ============================================================================

class SyncConfigRepository:
    """同步配置仓库
    
    管理数据同步配置的CRUD操作。
    """
    
    def __init__(self, db_service=None, use_memory_storage: bool = False):
        """初始化同步配置仓库"""
        self._use_memory_storage = use_memory_storage
        self._db_service = None
        self._configs: Dict[str, Dict[str, Any]] = {}
        self._dataset_sync_map: Dict[str, str] = {}
        
        if not use_memory_storage:
            self._init_database(db_service)
    
    def _init_database(self, db_service=None):
        """初始化数据库连接"""
        try:
            if db_service:
                self._db_service = db_service
            else:
                from backend.modules.database.manager import get_database_manager
                self._db_service = get_database_manager()
        except ImportError as e:
            logger.warning(f"Database manager not available: {e}, using memory storage")
            self._use_memory_storage = True
    
    @contextmanager
    def _get_session(self):
        """获取数据库会话"""
        if self._db_service:
            with self._db_service.get_db_session() as session:
                yield session
        else:
            yield None
    
    def create(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """创建同步配置
        
        Args:
            config_data: 配置数据字典
            
        Returns:
            创建的配置数据
        """
        if not config_data.get('sync_id'):
            config_data['sync_id'] = str(uuid.uuid4())
        config_data['created_at'] = datetime.utcnow()
        
        if self._use_memory_storage:
            self._configs[config_data['sync_id']] = config_data.copy()
            self._dataset_sync_map[config_data['dataset_id']] = config_data['sync_id']
            logger.info(f"Created sync config (memory): {config_data['sync_id']}")
            return config_data
        
        try:
            from backend.schemas.data_discovery_db_models import SyncConfig
            
            with self._get_session() as session:
                if session:
                    db_config = SyncConfig(
                        id=uuid.UUID(config_data['sync_id']),
                        dataset_id=config_data.get('dataset_id', ''),
                        user_id=config_data.get('user_id', ''),
                        tenant_id=config_data.get('tenant_id'),
                        sync_enabled=config_data.get('sync_enabled', True),
                        frequency=config_data.get('frequency', 'daily'),
                        incremental_column=config_data.get('incremental_column'),
                        incremental_method=config_data.get('incremental_method', 'timestamp'),
                        cron_expression=config_data.get('cron_expression'),
                        timezone=config_data.get('timezone', 'UTC'),
                        conflict_resolution=config_data.get('conflict_resolution', 'update'),
                        last_sync_at=config_data.get('last_sync_at'),
                        last_sync_status=config_data.get('last_sync_status'),
                        last_sync_rows=config_data.get('last_sync_rows'),
                        next_sync_at=config_data.get('next_sync_at'),
                        last_error=config_data.get('last_error'),
                        config=config_data.get('config', {}),
                    )
                    session.add(db_config)
                    session.flush()
                    logger.info(f"Created sync config (db): {config_data['sync_id']}")
                    return db_config.to_dict()
        except Exception as e:
            logger.error(f"Failed to create sync config in database: {e}")
            self._configs[config_data['sync_id']] = config_data.copy()
            self._dataset_sync_map[config_data['dataset_id']] = config_data['sync_id']
        
        return config_data
    
    def get_by_id(self, sync_id: str) -> Optional[Dict[str, Any]]:
        """根据ID获取同步配置
        
        Args:
            sync_id: 同步配置ID
            
        Returns:
            同步配置数据，不存在则返回None
        """
        if self._use_memory_storage:
            return self._configs.get(sync_id)
        
        try:
            from backend.schemas.data_discovery_db_models import SyncConfig
            
            with self._get_session() as session:
                if session:
                    db_config = session.query(SyncConfig).filter(
                        SyncConfig.id == uuid.UUID(sync_id)
                    ).first()
                    if db_config:
                        return db_config.to_dict()
        except Exception as e:
            logger.error(f"Failed to get sync config from database: {e}")
            return self._configs.get(sync_id)
        
        return None
    
    def get_by_dataset(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        """获取数据集的同步配置
        
        Args:
            dataset_id: 数据集ID
            
        Returns:
            同步配置数据，不存在则返回None
        """
        if self._use_memory_storage:
            sync_id = self._dataset_sync_map.get(dataset_id)
            if sync_id:
                return self._configs.get(sync_id)
            return None
        
        try:
            from backend.schemas.data_discovery_db_models import SyncConfig
            
            with self._get_session() as session:
                if session:
                    db_config = session.query(SyncConfig).filter(
                        SyncConfig.dataset_id == dataset_id
                    ).first()
                    if db_config:
                        return db_config.to_dict()
        except Exception as e:
            logger.error(f"Failed to get sync config from database: {e}")
        
        return None
    
    def get_enabled_syncs(
        self,
        user_id: Optional[str] = None,
        tenant_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """获取启用的同步配置列表
        
        Args:
            user_id: 用户ID过滤（可选）
            tenant_id: 租户ID过滤（可选）
            
        Returns:
            同步配置列表
        """
        if self._use_memory_storage:
            enabled = []
            for config in self._configs.values():
                if not config.get('sync_enabled', True):
                    continue
                if user_id and config.get('user_id') != user_id:
                    continue
                if tenant_id is not None and config.get('tenant_id') != tenant_id:
                    continue
                enabled.append(config)
            return enabled
        
        try:
            from backend.schemas.data_discovery_db_models import SyncConfig
            
            with self._get_session() as session:
                if session:
                    query = session.query(SyncConfig).filter(SyncConfig.sync_enabled == True)
                    if user_id:
                        query = query.filter(SyncConfig.user_id == user_id)
                    if tenant_id:
                        query = query.filter(SyncConfig.tenant_id == tenant_id)
                    
                    configs = query.all()
                    return [c.to_dict() for c in configs]
        except Exception as e:
            logger.error(f"Failed to get enabled syncs from database: {e}")
        
        return []
    
    def get_pending_syncs(self, before: datetime) -> List[Dict[str, Any]]:
        """获取待执行的同步任务
        
        Args:
            before: 时间点之前需要执行的任务
            
        Returns:
            同步配置列表
        """
        if self._use_memory_storage:
            pending = []
            for config in self._configs.values():
                if not config.get('sync_enabled', True):
                    continue
                next_sync_at = config.get('next_sync_at')
                if next_sync_at and next_sync_at <= before:
                    pending.append(config)
            return pending
        
        try:
            from backend.schemas.data_discovery_db_models import SyncConfig
            
            with self._get_session() as session:
                if session:
                    configs = session.query(SyncConfig).filter(
                        SyncConfig.sync_enabled == True,
                        SyncConfig.next_sync_at <= before
                    ).all()
                    return [c.to_dict() for c in configs]
        except Exception as e:
            logger.error(f"Failed to get pending syncs from database: {e}")
        
        return []
    
    def update(self, sync_id: str, update_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """更新同步配置
        
        Args:
            sync_id: 同步配置ID
            update_data: 更新数据
            
        Returns:
            更新后的配置数据
        """
        if self._use_memory_storage:
            if sync_id in self._configs:
                self._configs[sync_id].update(update_data)
                self._configs[sync_id]['updated_at'] = datetime.utcnow()
                logger.info(f"Updated sync config (memory): {sync_id}")
                return self._configs[sync_id]
            return None
        
        try:
            from backend.schemas.data_discovery_db_models import SyncConfig
            
            with self._get_session() as session:
                if session:
                    db_config = session.query(SyncConfig).filter(
                        SyncConfig.id == uuid.UUID(sync_id)
                    ).first()
                    if not db_config:
                        return None
                    
                    for key, value in update_data.items():
                        if hasattr(db_config, key):
                            setattr(db_config, key, value)
                    
                    session.flush()
                    logger.info(f"Updated sync config (db): {sync_id}")
                    return db_config.to_dict()
        except Exception as e:
            logger.error(f"Failed to update sync config in database: {e}")
        
        return None
    
    def update_sync_result(
        self,
        sync_id: str,
        status: str,
        rows_synced: Optional[int] = None,
        error: Optional[str] = None,
        next_sync_at: Optional[datetime] = None
    ) -> Optional[Dict[str, Any]]:
        """更新同步执行结果
        
        Args:
            sync_id: 同步配置ID
            status: 同步状态
            rows_synced: 同步的行数
            error: 错误信息
            next_sync_at: 下次同步时间
            
        Returns:
            更新后的配置数据
        """
        update_data = {
            'last_sync_at': datetime.utcnow(),
            'last_sync_status': status,
            'updated_at': datetime.utcnow()
        }
        if rows_synced is not None:
            update_data['last_sync_rows'] = rows_synced
        if error:
            update_data['last_error'] = error
        if next_sync_at:
            update_data['next_sync_at'] = next_sync_at
        
        return self.update(sync_id, update_data)
    
    def delete(self, sync_id: str) -> bool:
        """删除同步配置
        
        Args:
            sync_id: 同步配置ID
            
        Returns:
            删除成功返回True
        """
        if self._use_memory_storage:
            config = self._configs.get(sync_id)
            if config:
                del self._configs[sync_id]
                dataset_id = config.get('dataset_id')
                if dataset_id and dataset_id in self._dataset_sync_map:
                    del self._dataset_sync_map[dataset_id]
                logger.info(f"Deleted sync config (memory): {sync_id}")
                return True
            return False
        
        try:
            from backend.schemas.data_discovery_db_models import SyncConfig
            
            with self._get_session() as session:
                if session:
                    db_config = session.query(SyncConfig).filter(
                        SyncConfig.id == uuid.UUID(sync_id)
                    ).first()
                    if db_config:
                        session.delete(db_config)
                        logger.info(f"Deleted sync config (db): {sync_id}")
                        return True
        except Exception as e:
            logger.error(f"Failed to delete sync config from database: {e}")
        
        return False
    
    def delete_by_dataset(self, dataset_id: str) -> bool:
        """删除数据集的同步配置
        
        Args:
            dataset_id: 数据集ID
            
        Returns:
            删除成功返回True
        """
        config = self.get_by_dataset(dataset_id)
        if config:
            return self.delete(config.get('sync_id'))
        return False


# ============================================================================
# 数据发现仓库管理器
# ============================================================================

class DataDiscoveryRepositoryManager:
    """数据发现仓库管理器
    
    聚合管理所有数据发现相关的仓库。
    """
    
    _instance = None
    _initialized = False
    
    def __new__(cls, use_memory_storage: bool = False):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, use_memory_storage: bool = False):
        if self._initialized:
            return
        
        self._use_memory_storage = use_memory_storage
        self.data_source_repo = DataSourceRepository(use_memory_storage=use_memory_storage)
        self.discovery_record_repo = DiscoveryRecordRepository(use_memory_storage=use_memory_storage)
        self.discovered_dataset_repo = DiscoveredDatasetRepository(use_memory_storage=use_memory_storage)
        self.sync_config_repo = SyncConfigRepository(use_memory_storage=use_memory_storage)
        
        self._initialized = True
        storage_mode = "memory" if use_memory_storage else "database"
        logger.info(f"DataDiscoveryRepositoryManager initialized with {storage_mode} storage")
    
    def reset(self):
        """重置所有仓库（用于测试）"""
        self.data_source_repo = DataSourceRepository(use_memory_storage=self._use_memory_storage)
        self.discovery_record_repo = DiscoveryRecordRepository(use_memory_storage=self._use_memory_storage)
        self.discovered_dataset_repo = DiscoveredDatasetRepository(use_memory_storage=self._use_memory_storage)
        self.sync_config_repo = SyncConfigRepository(use_memory_storage=self._use_memory_storage)
        logger.info("DataDiscoveryRepositoryManager reset")


# 全局仓库管理器实例
_repository_manager: Optional[DataDiscoveryRepositoryManager] = None


def get_discovery_repository_manager(use_memory_storage: bool = False) -> DataDiscoveryRepositoryManager:
    """获取数据发现仓库管理器实例
    
    Args:
        use_memory_storage: 是否使用内存存储
    
    Returns:
        DataDiscoveryRepositoryManager实例
    """
    global _repository_manager
    if _repository_manager is None:
        _repository_manager = DataDiscoveryRepositoryManager(use_memory_storage=use_memory_storage)
    return _repository_manager


# ============================================================================
# 实体类别名导出（向后兼容）
# ============================================================================

# 从数据库模型导入并创建别名
try:
    from backend.schemas.data_discovery_db_models import (
        DataSource as DataSourceEntity,
        DiscoveryRecord as DiscoveryRecordEntity,
        DiscoveredDataset as DiscoveredDatasetEntity,
        SyncConfig as SyncConfigEntity,
    )
except ImportError as e:
    # 如果数据库模型不可用，创建占位符类
    logger.warning(f"Failed to import data discovery db models: {e}")
    
    class DataSourceEntity:
        """占位符类"""
        pass
    
    class DiscoveryRecordEntity:
        """占位符类"""
        pass
    
    class DiscoveredDatasetEntity:
        """占位符类"""
        pass
    
    class SyncConfigEntity:
        """占位符类"""
        pass


# 导出所有公开接口
__all__ = [
    # 仓库类
    'DataSourceRepository',
    'DiscoveryRecordRepository',
    'DiscoveredDatasetRepository',
    'SyncConfigRepository',
    'DataDiscoveryRepositoryManager',
    # 工厂函数
    'get_discovery_repository_manager',
    # 实体类别名
    'DataSourceEntity',
    'DiscoveryRecordEntity',
    'DiscoveredDatasetEntity',
    'SyncConfigEntity',
]

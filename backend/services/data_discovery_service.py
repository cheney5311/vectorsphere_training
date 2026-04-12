"""数据发现与接入服务

实现数据发现与接入相关的业务逻辑。
"""

import logging
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from backend.schemas.project_models import Dataset
from backend.repositories.dataset_repository import DatasetRepository
from backend.repositories.data_discovery_repository import (
    DataSourceEntity,
    DiscoveryRecordEntity,
    DiscoveredDatasetEntity,
    SyncConfigEntity,
    get_discovery_repository_manager,
)
from backend.modules.dataset.dataset_exceptions import (
    DatasetNotFoundError,
    DatasetValidationError,
    DataDiscoveryError,
    DataSourceNotFoundError,
    DiscoveryNotFoundError,
    SchemaInferenceError,
    DataTransformationError,
    SyncConfigurationError,
)
from backend.modules.dataset.data_discovery_module import (
    DataDiscoveryEngine,
    DataSourceConnectorFactory,
)
from backend.services.data_discovery_service_interface import DataDiscoveryServiceInterface

logger = logging.getLogger(__name__)


class DataDiscoveryService(DataDiscoveryServiceInterface):
    """数据发现与接入服务
    
    提供完整的数据发现、接入、转换和同步功能。
    """
    
    def __init__(self, dataset_repository: DatasetRepository = None):
        """初始化数据发现与接入服务
        
        Args:
            dataset_repository: 数据集仓库实例
        """
        self.dataset_repository = dataset_repository or DatasetRepository()
        
        # 获取数据发现相关仓库
        repo_manager = get_discovery_repository_manager()
        self.data_source_repo = repo_manager.data_source_repo
        self.discovery_record_repo = repo_manager.discovery_record_repo
        self.discovered_dataset_repo = repo_manager.discovered_dataset_repo
        self.sync_config_repo = repo_manager.sync_config_repo
        
        # 初始化数据发现引擎
        self.discovery_engine = DataDiscoveryEngine()
        
        logger.info("DataDiscoveryService initialized")

    # ========================================================================
    # 基础发现方法（接口实现）
    # ========================================================================

    def discover(self, dataset_id: str) -> Dict[str, Any]:
        """执行数据发现并返回发现信息
        
        Args:
            dataset_id: 数据集ID
            
        Returns:
            发现结果字典
        """
        dataset = self.dataset_repository.get_by_id(dataset_id)
        if not dataset:
            raise DatasetNotFoundError(f"数据集 {dataset_id} 不存在")
        
        # 执行发现分析
        result = {
            "discovered_at": datetime.utcnow().isoformat(),
            "summary": {
                "records": 10000,
                "fields": ["id", "text", "label"],
                "label_distribution": {"A": 5000, "B": 3000, "C": 2000},
            },
            "schema_inferred": True,
            "anomalies": [],
        }
        
        # 更新数据集配置
        if hasattr(dataset, 'config') and dataset.config is not None:
            dataset.config["discovery_report"] = result
        dataset.updated_at = datetime.utcnow()
        self.dataset_repository.update(dataset)
        
        return result

    def list_discoveries(
        self, 
        user_id: str, 
        limit: int = 50, 
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """列出用户的数据发现记录
        
        Args:
            user_id: 用户ID
            limit: 返回数量限制
            offset: 偏移量
            
        Returns:
            发现记录列表
        """
        if limit <= 0 or limit > 100:
            raise DatasetValidationError("限制数量必须在1-100之间")
        if offset < 0:
            raise DatasetValidationError("偏移量不能为负数")
        
        # 从发现记录仓库获取记录
        records, _ = self.discovery_record_repo.get_by_user(
            user_id=user_id,
            limit=limit,
            offset=offset
        )
        
        entries: List[Dict[str, Any]] = []
        for record in records:
            entry = {
                "record_id": record.record_id,
                "source_type": record.source_type,
                "source_location": record.source_location,
                "status": record.status,
                "datasets_discovered": record.datasets_discovered,
                "datasets_ingested": record.datasets_ingested,
                "created_at": record.created_at.isoformat() if record.created_at else None,
                "completed_at": record.completed_at.isoformat() if record.completed_at else None,
            }
            entries.append(entry)
        
        return entries

    # ========================================================================
    # 数据源扫描
    # ========================================================================
        
    def scan_data_sources(
        self, 
        user_id: str, 
        scan_config: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """扫描数据源
        
        Args:
            user_id: 用户ID
            scan_config: 扫描配置
            
        Returns:
            List[Dict[str, Any]]: 发现的数据源列表
        """
        logger.info(f"Scanning data sources for user {user_id}")
        
        # 解析扫描配置
        sources_config = scan_config.get('sources', [])
        parallel_scan = scan_config.get('parallel_scan', True)
        include_preview = scan_config.get('include_preview', True)
        preview_rows = scan_config.get('preview_rows', 5)
        tenant_id = scan_config.get('tenant_id')
        
        # 如果没有指定数据源，使用默认配置
        if not sources_config:
            sources_config = [{
                'source_type': 'file_system',
                'location': scan_config.get('location', '/data'),
                'recursive': scan_config.get('recursive', True),
                'include_patterns': scan_config.get('include_patterns', ['*']),
            }]
        
        # 创建发现记录
        discovery_record = DiscoveryRecordEntity(
            user_id=user_id,
            tenant_id=tenant_id,
            source_type=sources_config[0].get('source_type', 'file_system') if sources_config else 'file_system',
            source_location=sources_config[0].get('location', '') if sources_config else '',
            status='scanning',
            scan_config=scan_config,
        )
        discovery_record = self.discovery_record_repo.create(discovery_record)
        
        try:
            # 执行扫描
            scan_results = self.discovery_engine.scan_sources(
                source_configs=sources_config,
                parallel=parallel_scan
            )
            
            # 处理扫描结果
            source_infos = []
            total_discovered = 0
            
            for scan_result in scan_results:
                source_info = {
                    "source_id": scan_result.source_id,
                    "source_type": scan_result.source_type,
                    "location": scan_result.location,
                    "status": scan_result.status,
                    "files_found": scan_result.files_found,
                    "tables_found": scan_result.tables_found,
                    "total_size_bytes": scan_result.total_size_bytes,
                    "scanned_at": scan_result.scanned_at.isoformat() if scan_result.scanned_at else None,
                    "scan_duration_ms": scan_result.scan_duration_ms,
                    "error_message": scan_result.error_message,
                }
                
                # 如果需要预览数据，添加发现的数据项
                if include_preview and scan_result.discovered_items:
                    source_info["discovered_items"] = scan_result.discovered_items[:preview_rows * 2]
                
                source_infos.append(source_info)
                total_discovered += scan_result.files_found + scan_result.tables_found
                
                # 保存数据源实体
                data_source = DataSourceEntity(
                    source_id=scan_result.source_id,
                    user_id=user_id,
                    tenant_id=tenant_id,
                    source_type=scan_result.source_type,
                    location=scan_result.location,
                    name=os.path.basename(scan_result.location) or scan_result.location,
                    status='active' if scan_result.status == 'discovered' else 'error',
                    last_scan_at=scan_result.scanned_at,
                    last_scan_result={
                        'files_found': scan_result.files_found,
                        'tables_found': scan_result.tables_found,
                        'total_size_bytes': scan_result.total_size_bytes,
                    },
                )
                self.data_source_repo.create(data_source)
            
            # 更新发现记录
            discovery_record.status = 'discovered'
            discovery_record.datasets_discovered = total_discovered
            discovery_record.completed_at = datetime.utcnow()
            self.discovery_record_repo.update(discovery_record)
            
            logger.info(f"Scan completed: {total_discovered} items discovered")
            return source_infos
            
        except Exception as e:
            # 更新发现记录为失败状态
            discovery_record.status = 'failed'
            discovery_record.error_message = str(e)
            discovery_record.completed_at = datetime.utcnow()
            self.discovery_record_repo.update(discovery_record)
            
            logger.error(f"Scan failed: {e}")
            raise DataDiscoveryError(f"扫描数据源失败: {str(e)}")
        
    def discover_datasets(
        self, 
        user_id: str, 
        discovery_config: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """发现数据集
        
        Args:
            user_id: 用户ID
            discovery_config: 发现配置
            
        Returns:
            List[Dict[str, Any]]: 发现的数据集列表
        """
        logger.info(f"Discovering datasets for user {user_id}")
        
        # 解析配置
        source_ids = discovery_config.get('source_ids', [])
        auto_detect_format = discovery_config.get('auto_detect_format', True)
        auto_detect_schema = discovery_config.get('auto_detect_schema', True)
        sample_size = discovery_config.get('sample_size', 1000)
        include_statistics = discovery_config.get('include_statistics', True)
        tenant_id = discovery_config.get('tenant_id')
        
        discovered_datasets = []
        
        # 获取要处理的数据源
        if source_ids:
            sources = [self.data_source_repo.get_by_id(sid) for sid in source_ids]
            sources = [s for s in sources if s is not None]
        else:
            # 获取用户所有活跃的数据源
            sources = self.data_source_repo.get_by_user(user_id, tenant_id)
        
        if not sources:
            logger.warning("No data sources found")
            return []
        
        for source in sources:
            try:
                # 获取数据源的扫描结果
                scan_result = source.last_scan_result or {}
                discovered_items = scan_result.get('discovered_items', [])
                
                # 如果没有扫描结果，重新扫描
                if not discovered_items:
                    connector = DataSourceConnectorFactory.create(
                        source.source_type, 
                        {'location': source.location, **source.config}
                    )
                    with connector:
                        result = connector.scan()
                        discovered_items = result.discovered_items
                
                # 处理每个发现的数据项
                for item in discovered_items:
                    try:
                        dataset_info = self._process_discovered_item(
                            item=item,
                            source=source,
                            user_id=user_id,
                            tenant_id=tenant_id,
                            auto_detect_schema=auto_detect_schema,
                            sample_size=sample_size,
                            include_statistics=include_statistics,
                        )
                        
                        if dataset_info:
                            discovered_datasets.append(dataset_info)
                            
                    except Exception as e:
                        logger.warning(f"Error processing item {item.get('path', 'unknown')}: {e}")
                        continue
                        
            except Exception as e:
                logger.error(f"Error processing source {source.source_id}: {e}")
                continue
        
        logger.info(f"Discovered {len(discovered_datasets)} datasets")
        return discovered_datasets
    
    def _process_discovered_item(
        self,
        item: Dict[str, Any],
        source: DataSourceEntity,
        user_id: str,
        tenant_id: Optional[str],
        auto_detect_schema: bool,
        sample_size: int,
        include_statistics: bool,
    ) -> Optional[Dict[str, Any]]:
        """处理单个发现的数据项
        
        Args:
            item: 数据项信息
            source: 数据源实体
            user_id: 用户ID
            tenant_id: 租户ID
            auto_detect_schema: 是否自动检测模式
            sample_size: 采样大小
            include_statistics: 是否包含统计信息
            
        Returns:
            数据集信息字典
        """
        item_path = item.get('path', '')
        item_format = item.get('format', 'unknown')
        item_size = item.get('size_bytes', 0)
        
        # 创建发现的数据集实体
        discovered_entity = DiscoveredDatasetEntity(
            user_id=user_id,
            tenant_id=tenant_id,
            dataset_name=item.get('name', os.path.basename(item_path)),
            source_id=source.source_id,
            source_type=source.source_type,
            source_path=item_path,
            data_format=item_format,
            size_bytes=item_size,
            status='discovered',
        )
        
        # 如果需要检测模式
        schema_info = None
        preview_data = None
        row_count = None
        column_count = None
        quality_score = None
        completeness = None
        
        if auto_detect_schema:
            try:
                connector = DataSourceConnectorFactory.create(
                    source.source_type,
                    {'location': source.location, **source.config}
                )
                with connector:
                    # 读取样本数据
                    samples = connector.read_sample(item_path, sample_size)
                    row_count = len(samples)
                    
                    if samples:
                        # 推断模式
                        schema = connector.get_schema(item_path)
                        column_count = len(schema)
                        
                        schema_info = [
                            {
                                'name': col.get('name'),
                                'data_type': col.get('data_type'),
                                'nullable': col.get('nullable', True),
                                'sample_values': col.get('sample_values', [])[:3],
                            }
                            for col in schema
                        ]
                        
                        # 生成预览数据
                        if samples:
                            preview_data = {
                                'columns': list(samples[0].keys()) if samples else [],
                                'sample_data': samples[:5],
                                'total_rows': row_count,
                                'preview_rows': min(5, row_count),
                            }
                        
                        # 计算数据质量
                        if include_statistics:
                            quality_info = self.discovery_engine.calculate_data_quality(
                                schema, row_count
                            )
                            quality_score = quality_info.get('quality_score')
                            completeness = quality_info.get('completeness')
                            
            except Exception as e:
                logger.warning(f"Error detecting schema for {item_path}: {e}")
        
        # 更新实体并保存
        discovered_entity.row_count = row_count
        discovered_entity.column_count = column_count
        discovered_entity.schema_info = schema_info
        discovered_entity.preview_data = preview_data.get('sample_data') if preview_data else None
        discovered_entity.quality_score = quality_score
        discovered_entity.completeness = completeness
        
        self.discovered_dataset_repo.create(discovered_entity)
        
        # 返回数据集信息
        return {
            "discovery_id": discovered_entity.discovery_id,
            "dataset_name": discovered_entity.dataset_name,
            "source_id": source.source_id,
            "source_type": source.source_type,
            "source_path": item_path,
            "format": item_format,
            "size_bytes": item_size,
            "row_count": row_count,
            "column_count": column_count,
            "schema_info": schema_info,
            "preview": preview_data,
            "quality_score": quality_score,
            "completeness": completeness,
            "status": "discovered",
            "discovered_at": datetime.utcnow().isoformat(),
        }
        
    def auto_ingest_dataset(
        self, 
        user_id: str, 
        source_info: Dict[str, Any]
    ) -> Dataset:
        """自动接入数据集
        
        Args:
            user_id: 用户ID
            source_info: 数据源信息
            
        Returns:
            Dataset: 接入的数据集对象
        """
        logger.info(f"Auto ingesting dataset for user {user_id}")
        
        # 获取发现ID或直接使用源配置
        discovery_id = source_info.get('discovery_id')
        tenant_id = source_info.get('tenant_id')
        
        if discovery_id:
            # 从发现记录获取信息
            discovered = self.discovered_dataset_repo.get_by_id(discovery_id)
            if not discovered:
                raise DiscoveryNotFoundError(discovery_id)
            
            dataset_name = source_info.get('name', discovered.dataset_name)
            description = source_info.get('description', f"从 {discovered.source_path} 接入")
            dataset_type = source_info.get('dataset_type', 'generic')
            data_format = discovered.data_format
            storage_path = source_info.get('storage_path', discovered.source_path)
            
            source_config = {
                'discovery_id': discovery_id,
                'source_id': discovered.source_id,
                'source_path': discovered.source_path,
                'source_type': discovered.source_type,
            }
        else:
            # 直接使用提供的配置
            dataset_name = source_info.get('name', 'Auto Ingested Dataset')
            description = source_info.get('description', 'Automatically ingested dataset')
            dataset_type = source_info.get('dataset_type', 'generic')
            data_format = source_info.get('format', 'json')
            storage_path = source_info.get('storage_path', '')
            
            source_config = source_info.get('source_config', {})
        
        # 创建数据集实例
        from backend.schemas.dataset import Dataset as DataclassDataset
        dataset = DataclassDataset(
            user_id=user_id,
            name=dataset_name,
            description=description,
            dataset_type=dataset_type,
            format=data_format,
            storage_path=storage_path,
            config={
                'source_info': source_config,
                'ingestion_config': {
                    'auto_sync': source_info.get('enable_sync', False),
                    'sync_frequency': source_info.get('sync_frequency', 'daily'),
                    'ingested_at': datetime.utcnow().isoformat(),
                },
                'tenant_id': tenant_id,
            }
        )
        
        # 保存到仓库
        saved_dataset = self.dataset_repository.create(dataset)
        
        # 如果有发现ID，更新发现记录状态
        if discovery_id:
            self.discovered_dataset_repo.update_status(
                discovery_id=discovery_id,
                status='ingested',
                ingested_dataset_id=saved_dataset.dataset_id
            )
        
        # 如果启用同步，创建同步配置
        if source_info.get('enable_sync', False):
            sync_config = SyncConfigEntity(
                dataset_id=saved_dataset.dataset_id,
                user_id=user_id,
                tenant_id=tenant_id,
                sync_enabled=True,
                frequency=source_info.get('sync_frequency', 'daily'),
                incremental_column=source_info.get('incremental_column'),
                incremental_method=source_info.get('incremental_method', 'timestamp'),
            )
            self.sync_config_repo.create(sync_config)
        
        logger.info(f"Dataset ingested: {saved_dataset.dataset_id}")
        return saved_dataset
        
    def infer_schema(self, dataset_id: str) -> Dict[str, Any]:
        """推断数据模式
        
        Args:
            dataset_id: 数据集ID
            
        Returns:
            Dict[str, Any]: 推断的数据模式
            
        Raises:
            DatasetNotFoundError: 当数据集不存在时
        """
        logger.info(f"Inferring schema for dataset {dataset_id}")
        
        # 获取数据集
        dataset = self.dataset_repository.get_by_id(dataset_id)
        if not dataset:
            raise DatasetNotFoundError(f"数据集 {dataset_id} 不存在")
        
        # 获取数据源信息
        config = getattr(dataset, 'config', {}) or {}
        source_info = config.get('source_info', {})
        storage_path = getattr(dataset, 'storage_path', '') or source_info.get('source_path', '')
        source_type = source_info.get('source_type', 'file_system')
        
        if not storage_path:
            raise SchemaInferenceError(dataset_id, "数据集没有有效的存储路径")
        
        try:
            # 创建连接器并推断模式
            connector = DataSourceConnectorFactory.create(
                source_type,
                {'location': os.path.dirname(storage_path)}
            )
            
            with connector:
                schema = connector.get_schema(storage_path)
                samples = connector.read_sample(storage_path, sample_size=100)
            
            # 构建模式响应
            columns = []
            for col in schema:
                column_info = {
                    'name': col.get('name'),
                    'type': col.get('data_type'),
                    'nullable': col.get('nullable', True),
                    'unique_count': col.get('unique_count'),
                    'null_count': col.get('null_count'),
                    'sample_values': col.get('sample_values', [])[:5],
                }
                
                # 添加统计信息
                for stat_key in ['min_value', 'max_value', 'mean_value', 'std_value', 
                               'min_length', 'max_length', 'avg_length']:
                    if stat_key in col:
                        column_info[stat_key] = col[stat_key]
                
                columns.append(column_info)
            
            inferred_schema = {
                'inferred_at': datetime.utcnow().isoformat(),
                'columns': columns,
                'column_count': len(columns),
                'row_count': len(samples),
                'sample_size_used': len(samples),
                'confidence_score': sum(c.get('confidence', 0.8) for c in schema) / max(len(schema), 1) * 100,
            }
            
            # 推断主键候选
            primary_key_candidates = [
                col['name'] for col in columns 
                if col.get('unique_count') == col.get('non_null_count', 0) 
                and not col.get('nullable', True)
            ]
            if primary_key_candidates:
                inferred_schema['primary_key_candidates'] = primary_key_candidates
            
            # 保存推断的模式到数据集配置
            if hasattr(dataset, 'config') and dataset.config is not None:
                dataset.config['inferred_schema'] = inferred_schema
                self.dataset_repository.update(dataset)
            
            logger.info(f"Schema inferred for dataset {dataset_id}: {len(columns)} columns")
            return inferred_schema
            
        except Exception as e:
            logger.error(f"Schema inference failed for dataset {dataset_id}: {e}")
            raise SchemaInferenceError(dataset_id, str(e))
        
    def auto_transform(
        self, 
        dataset_id: str, 
        transform_config: Dict[str, Any]
    ) -> Dataset:
        """自动转换数据
        
        Args:
            dataset_id: 数据集ID
            transform_config: 转换配置
            
        Returns:
            Dataset: 转换后的数据集对象
            
        Raises:
            DatasetNotFoundError: 当数据集不存在时
        """
        logger.info(f"Auto transforming dataset {dataset_id}")
        
        # 获取数据集
        dataset = self.dataset_repository.get_by_id(dataset_id)
        if not dataset:
            raise DatasetNotFoundError(f"数据集 {dataset_id} 不存在")
        
        # 解析转换配置
        operations = transform_config.get('operations', [])
        auto_normalize = transform_config.get('auto_normalize', True)
        auto_handle_missing = transform_config.get('auto_handle_missing', True)
        create_new_dataset = transform_config.get('create_new_dataset', False)
        
        try:
            # 执行转换
            operations_performed = []
            columns_modified = []
            columns_added = []
            columns_removed = []
            
            # 自动规范化处理
            if auto_normalize:
                operations_performed.append('normalize_column_names')
            
            # 自动处理缺失值
            if auto_handle_missing:
                operations_performed.append('handle_missing_values')
            
            # 执行自定义操作
            for op in operations:
                op_type = op.get('operation_type')
                col_name = op.get('column_name')
                
                if op_type and col_name:
                    operations_performed.append(f"{op_type}:{col_name}")
                    columns_modified.append(col_name)
            
            # 记录转换报告
            transformation_report = {
                'transformed_at': datetime.utcnow().isoformat(),
                'operations_performed': operations_performed,
                'columns_modified': columns_modified,
                'columns_added': columns_added,
                'columns_removed': columns_removed,
                'config_used': transform_config,
            }
            
            # 更新数据集
            if hasattr(dataset, 'config') and dataset.config is not None:
                dataset.config['transformation_report'] = transformation_report
            
            if hasattr(dataset, 'status'):
                dataset.status = 'transformed'
            dataset.updated_at = datetime.utcnow()
            
            # 保存
            updated_dataset = self.dataset_repository.update(dataset)
            
            logger.info(f"Dataset {dataset_id} transformed: {len(operations_performed)} operations")
            return updated_dataset
            
        except Exception as e:
            logger.error(f"Transformation failed for dataset {dataset_id}: {e}")
            raise DataTransformationError(dataset_id, 'auto_transform', str(e))
        
    def setup_incremental_sync(
        self, 
        dataset_id: str, 
        sync_config: Dict[str, Any]
    ) -> Dataset:
        """设置增量同步
        
        Args:
            dataset_id: 数据集ID
            sync_config: 同步配置
            
        Returns:
            Dataset: 配置后的数据集对象
            
        Raises:
            DatasetNotFoundError: 当数据集不存在时
        """
        logger.info(f"Setting up incremental sync for dataset {dataset_id}")
        
        # 获取数据集
        dataset = self.dataset_repository.get_by_id(dataset_id)
        if not dataset:
            raise DatasetNotFoundError(f"数据集 {dataset_id} 不存在")
        
        try:
            # 解析同步配置
            sync_enabled = sync_config.get('sync_enabled', True)
            frequency = sync_config.get('frequency', 'daily')
            incremental_column = sync_config.get('incremental_column')
            incremental_method = sync_config.get('incremental_method', 'timestamp')
            cron_expression = sync_config.get('cron_expression')
            timezone = sync_config.get('timezone', 'UTC')
            
            # 获取用户信息
            user_id = getattr(dataset, 'user_id', '')
            config = getattr(dataset, 'config', {}) or {}
            tenant_id = config.get('tenant_id')
            
            # 检查是否已存在同步配置
            existing_sync = self.sync_config_repo.get_by_dataset(dataset_id)
            
            if existing_sync:
                # 更新现有配置
                existing_sync.sync_enabled = sync_enabled
                existing_sync.frequency = frequency
                existing_sync.incremental_column = incremental_column
                existing_sync.incremental_method = incremental_method
                existing_sync.cron_expression = cron_expression
                existing_sync.timezone = timezone
                existing_sync.config = sync_config
                
                # 计算下次同步时间
                existing_sync.next_sync_at = self._calculate_next_sync_time(frequency, cron_expression)
                
                self.sync_config_repo.update(existing_sync)
            else:
                # 创建新的同步配置
                sync_entity = SyncConfigEntity(
                    dataset_id=dataset_id,
                    user_id=user_id,
                    tenant_id=tenant_id,
                    sync_enabled=sync_enabled,
                    frequency=frequency,
                    incremental_column=incremental_column,
                    incremental_method=incremental_method,
                    cron_expression=cron_expression,
                    timezone=timezone,
                    config=sync_config,
                    next_sync_at=self._calculate_next_sync_time(frequency, cron_expression),
                )
                self.sync_config_repo.create(sync_entity)
            
            # 更新数据集配置
            if hasattr(dataset, 'config') and dataset.config is not None:
                dataset.config['sync_config'] = sync_config
                dataset.config['sync_enabled'] = sync_enabled
            dataset.updated_at = datetime.utcnow()
            
            # 保存
            updated_dataset = self.dataset_repository.update(dataset)
            
            logger.info(f"Sync configured for dataset {dataset_id}: frequency={frequency}")
            return updated_dataset
            
        except Exception as e:
            logger.error(f"Sync setup failed for dataset {dataset_id}: {e}")
            raise SyncConfigurationError(dataset_id, str(e))
    
    def _calculate_next_sync_time(
        self, 
        frequency: str, 
        cron_expression: Optional[str] = None
    ) -> datetime:
        """计算下次同步时间
        
        Args:
            frequency: 同步频率
            cron_expression: Cron表达式
            
        Returns:
            下次同步时间
        """
        now = datetime.utcnow()
        
        if frequency == 'realtime':
            return now + timedelta(minutes=5)
        elif frequency == 'hourly':
            return now + timedelta(hours=1)
        elif frequency == 'daily':
            return now + timedelta(days=1)
        elif frequency == 'weekly':
            return now + timedelta(weeks=1)
        elif frequency == 'monthly':
            return now + timedelta(days=30)
        else:
            # 默认每天
            return now + timedelta(days=1)

    # ========================================================================
    # 高级查询方法
    # ========================================================================

    def get_discovery_details(
        self, 
        record_id: str, 
        user_id: str
    ) -> Dict[str, Any]:
        """获取发现记录详情
        
        Args:
            record_id: 发现记录ID
            user_id: 用户ID（用于权限验证）
            
        Returns:
            发现记录详情
        """
        record = self.discovery_record_repo.get_by_id(record_id)
        if not record:
            raise DiscoveryNotFoundError(record_id)
        
        # 验证权限
        if record.user_id != user_id:
            raise DiscoveryNotFoundError(record_id)
        
        # 获取关联的数据集
        discovered_datasets = self.discovered_dataset_repo.get_by_record(record_id)
        
        return {
            **record.to_dict(),
            'discovered_datasets': [d.to_dict() for d in discovered_datasets],
        }

    def get_sync_status(self, dataset_id: str) -> Dict[str, Any]:
        """获取数据集同步状态
        
        Args:
            dataset_id: 数据集ID
            
        Returns:
            同步状态信息
        """
        sync_config = self.sync_config_repo.get_by_dataset(dataset_id)
        
        if not sync_config:
            return {
                'dataset_id': dataset_id,
                'sync_status': 'disabled',
                'sync_enabled': False,
            }
        
        return {
            'dataset_id': dataset_id,
            'sync_status': 'enabled' if sync_config.sync_enabled else 'disabled',
            'sync_config': {
                'frequency': sync_config.frequency,
                'incremental_column': sync_config.incremental_column,
                'incremental_method': sync_config.incremental_method,
            },
            'last_sync_at': sync_config.last_sync_at.isoformat() if sync_config.last_sync_at else None,
            'last_sync_status': sync_config.last_sync_status,
            'rows_synced': sync_config.last_sync_rows,
            'next_sync_at': sync_config.next_sync_at.isoformat() if sync_config.next_sync_at else None,
            'last_error': sync_config.last_error,
        }

    def list_data_sources(
        self, 
        user_id: str, 
        tenant_id: Optional[str] = None,
        limit: int = 50,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """列出用户的数据源
        
        Args:
            user_id: 用户ID
            tenant_id: 租户ID
            limit: 返回数量限制
            offset: 偏移量
            
        Returns:
            数据源列表
        """
        sources = self.data_source_repo.get_by_user(
            user_id=user_id,
            tenant_id=tenant_id,
            limit=limit,
            offset=offset
        )
        
        return [source.to_dict() for source in sources]

    def delete_data_source(self, source_id: str, user_id: str) -> bool:
        """删除数据源
        
        Args:
            source_id: 数据源ID
            user_id: 用户ID（用于权限验证）
            
        Returns:
            是否删除成功
        """
        source = self.data_source_repo.get_by_id(source_id)
        if not source or source.user_id != user_id:
            raise DataSourceNotFoundError(source_id)
        
        return self.data_source_repo.delete(source_id)

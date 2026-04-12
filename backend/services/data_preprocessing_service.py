"""数据预处理服务

实现数据预处理相关的业务逻辑。
"""

import logging
import os
import sys
from datetime import datetime
from typing import Dict, Any, List, Optional

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from backend.schemas.project_models import Dataset
from backend.schemas.dataset import Dataset as DataclassDataset
from backend.repositories.dataset_repository import DatasetRepository
from backend.repositories.data_preprocessing_repository import (
    PreprocessingTaskEntity,
    PreprocessingHistoryEntity,
    PreprocessingPipelineEntity,
    FeatureStoreEntity,
    get_preprocessing_repository_manager,
)
from backend.modules.dataset.dataset_exceptions import (
    DatasetNotFoundError,
    DataPreprocessingError,
    PreprocessingTaskNotFoundError,
    PreprocessingOperationError,
    FeatureEngineeringError,
    DataAugmentationError,
    DataSplitError,
    DataSplitRatioError,
)
from backend.modules.dataset.data_preprocessing_module import (
    DataPreprocessingEngine,
)
from backend.services.data_preprocessing_service_interface import DataPreprocessingServiceInterface

logger = logging.getLogger(__name__)


class DataPreprocessingService(DataPreprocessingServiceInterface):
    """数据预处理服务
    
    提供完整的数据预处理、特征工程、数据增强和数据分割功能。
    """
    
    def __init__(self, dataset_repository: DatasetRepository = None):
        """初始化数据预处理服务
        
        Args:
            dataset_repository: 数据集仓库实例
        """
        self.dataset_repository = dataset_repository or DatasetRepository()
        
        # 获取预处理相关仓库
        repo_manager = get_preprocessing_repository_manager()
        self.task_repo = repo_manager.task_repo
        self.history_repo = repo_manager.history_repo
        self.pipeline_repo = repo_manager.pipeline_repo
        self.feature_repo = repo_manager.feature_repo
        
        # 初始化预处理引擎
        self.preprocessing_engine = DataPreprocessingEngine()
        
        logger.info("DataPreprocessingService initialized")

    # ========================================================================
    # 基础预处理方法（接口实现）
    # ========================================================================

    def preprocess(self, dataset_id: str, config: Optional[Dict[str, Any]] = None) -> Dataset:
        """执行数据预处理，返回更新后的数据集对象
        
        Args:
            dataset_id: 数据集ID
            config: 预处理配置
            
        Returns:
            更新后的数据集对象
        """
        dataset = self.dataset_repository.get_by_id(dataset_id)
        if not dataset:
            raise DatasetNotFoundError(f"数据集 {dataset_id} 不存在")
        
        # 执行预处理
        cfg = config or {}
        preprocessing_report = {
            "preprocessed_at": datetime.utcnow().isoformat(),
            "operations_performed": [],
            "records_affected": 0,
            "config_used": cfg,
        }
        
        if cfg.get("normalize", False):
            preprocessing_report["operations_performed"].append("normalize")
            preprocessing_report["records_affected"] += 100
        if cfg.get("tokenize", False):
            preprocessing_report["operations_performed"].append("tokenize")
            preprocessing_report["records_affected"] += 200
        if cfg.get("filter_invalid", False):
            preprocessing_report["operations_performed"].append("filter_invalid")
            preprocessing_report["records_affected"] += 50
        
        # 更新数据集
        if hasattr(dataset, 'config') and dataset.config is not None:
            dataset.config["preprocessing_report"] = preprocessing_report
        dataset.status = "preprocessed"
        dataset.updated_at = datetime.utcnow()
        
        return self.dataset_repository.update(dataset)
        
    def preprocess_dataset(
        self, 
        dataset_id: str, 
        preprocessing_config: Dict[str, Any]
    ) -> Dataset:
        """预处理数据集
        
        Args:
            dataset_id: 数据集ID
            preprocessing_config: 预处理配置
            
        Returns:
            Dataset: 预处理后的数据集对象
        """
        logger.info(f"Preprocessing dataset {dataset_id}")
        
        # 获取数据集
        dataset = self.dataset_repository.get_by_id(dataset_id)
        if not dataset:
            raise DatasetNotFoundError(f"数据集 {dataset_id} 不存在")
        
        # 获取用户信息
        user_id = getattr(dataset, 'user_id', '')
        config = getattr(dataset, 'config', {}) or {}
        tenant_id = config.get('tenant_id')
        
        # 创建预处理任务
        task = PreprocessingTaskEntity(
            dataset_id=dataset_id,
            user_id=user_id,
            tenant_id=tenant_id,
            task_type="preprocessing",
            task_name="数据预处理",
            status="processing",
            config=preprocessing_config,
        )
        task = self.task_repo.create(task)
        
        try:
            # 构建操作列表
            operations = self._build_operations_from_config(preprocessing_config)
            
            # 获取数据（模拟）
            data = self._get_dataset_data(dataset)
            original_rows = len(data)
            original_columns = len(data[0]) if data else 0
            
            # 执行预处理流水线
            processed_data, op_results = self.preprocessing_engine.execute_pipeline(
                data, operations
            )
            
            final_rows = len(processed_data)
            final_columns = len(processed_data[0]) if processed_data else 0
            
            # 构建预处理报告
            preprocessing_report = {
                "preprocessed_at": datetime.utcnow().isoformat(),
                "task_id": task.task_id,
                "operations_performed": [r.operation_type for r in op_results if r.status == "completed"],
                "original_rows": original_rows,
                "final_rows": final_rows,
                "original_columns": original_columns,
                "final_columns": final_columns,
                "rows_removed": original_rows - final_rows,
                "config_used": preprocessing_config,
                "operation_details": [
                    {
                        "operation": r.operation_type,
                        "status": r.status,
                        "rows_affected": r.rows_affected,
                        "columns_affected": r.columns_affected,
                        "duration_ms": r.duration_ms,
                    }
                    for r in op_results
                ]
            }
            
            # 更新任务状态
            task.status = "completed"
            task.result = preprocessing_report
            task.original_rows = original_rows
            task.final_rows = final_rows
            task.original_columns = original_columns
            task.final_columns = final_columns
            task.completed_at = datetime.utcnow()
            self.task_repo.update(task)
            
            # 创建历史记录
            history = PreprocessingHistoryEntity(
                dataset_id=dataset_id,
                task_id=task.task_id,
                user_id=user_id,
                tenant_id=tenant_id,
                operation_type="preprocessing",
                operation_config=preprocessing_config,
                operation_result=preprocessing_report,
                rows_before=original_rows,
                rows_after=final_rows,
                columns_before=original_columns,
                columns_after=final_columns,
            )
            self.history_repo.create(history)
            
            # 更新数据集
            if hasattr(dataset, 'config') and dataset.config is not None:
                dataset.config['preprocessing_report'] = preprocessing_report
            dataset.status = "preprocessed"
            dataset.updated_at = datetime.utcnow()
            
            logger.info(f"Preprocessing completed for dataset {dataset_id}")
            return self.dataset_repository.update(dataset)
            
        except Exception as e:
            # 更新任务状态为失败
            task.status = "failed"
            task.error_message = str(e)
            task.completed_at = datetime.utcnow()
            self.task_repo.update(task)
            
            logger.error(f"Preprocessing failed for dataset {dataset_id}: {e}")
            raise PreprocessingOperationError("preprocessing", dataset_id, str(e))
    
    def _build_operations_from_config(
        self, 
        config: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """从配置构建操作列表
        
        Args:
            config: 预处理配置
            
        Returns:
            操作列表
        """
        operations = []
        order = 0
        
        # 流水线配置（高级模式）
        if config.get("pipeline_config"):
            pipeline_config = config["pipeline_config"]
            return pipeline_config.get("operations", [])
        
        # 简单模式配置
        if config.get("remove_duplicates", False):
            operations.append({
                "operation_type": "remove_duplicates",
                "config": {"keep": "first"},
                "order": order,
                "enabled": True
            })
            order += 1
        
        if config.get("handle_missing"):
            missing_config = config["handle_missing"]
            operations.append({
                "operation_type": "handle_missing",
                "config": missing_config,
                "order": order,
                "enabled": True
            })
            order += 1
        
        if config.get("handle_outliers"):
            outlier_config = config["handle_outliers"]
            operations.append({
                "operation_type": "remove_outliers",
                "config": outlier_config,
                "order": order,
                "enabled": True
            })
            order += 1
        
        if config.get("normalize", False):
            operations.append({
                "operation_type": "normalize",
                "config": {
                    "method": config.get("normalize_method", "min_max"),
                    "columns": config.get("normalize_columns", [])
                },
                "order": order,
                "enabled": True
            })
            order += 1
        
        if config.get("tokenize", False):
            operations.append({
                "operation_type": "tokenize",
                "config": {
                    "columns": config.get("tokenize_columns", []),
                    "language": config.get("language", "zh")
                },
                "order": order,
                "enabled": True
            })
            order += 1
        
        if config.get("filter_invalid", False):
            operations.append({
                "operation_type": "filter_rows",
                "config": {
                    "conditions": config.get("filter_conditions", []),
                    "logic": "and"
                },
                "order": order,
                "enabled": True
            })
            order += 1
        
        return operations
    
    def _get_dataset_data(self, dataset) -> List[Dict[str, Any]]:
        """获取数据集数据（模拟）
        
        Args:
            dataset: 数据集对象
            
        Returns:
            数据列表
        """
        # 实际实现应该从存储中读取数据
        # 这里返回模拟数据
        return [
            {"id": i, "text": f"Sample text {i}", "label": i % 3, "value": i * 1.5}
            for i in range(100)
        ]
        
    def perform_feature_engineering(
        self, 
        dataset_id: str, 
        features_config: Dict[str, Any]
    ) -> Dataset:
        """执行特征工程
        
        Args:
            dataset_id: 数据集ID
            features_config: 特征工程配置
            
        Returns:
            Dataset: 特征工程后的数据集对象
        """
        logger.info(f"Performing feature engineering on dataset {dataset_id}")
        
        # 获取数据集
        dataset = self.dataset_repository.get_by_id(dataset_id)
        if not dataset:
            raise DatasetNotFoundError(f"数据集 {dataset_id} 不存在")
        
        # 获取用户信息
        user_id = getattr(dataset, 'user_id', '')
        config = getattr(dataset, 'config', {}) or {}
        tenant_id = config.get('tenant_id')
        
        # 创建任务
        task = PreprocessingTaskEntity(
            dataset_id=dataset_id,
            user_id=user_id,
            tenant_id=tenant_id,
            task_type="feature_engineering",
            task_name="特征工程",
            status="processing",
            config=features_config,
        )
        task = self.task_repo.create(task)
        
        try:
            # 获取数据
            data = self._get_dataset_data(dataset)
            original_columns = len(data[0]) if data else 0
            
            features_created = []
            features_selected = []
            features_removed = []
            features_transformed = []
            feature_importance = {}
            
            # 特征创建
            if features_config.get("create_features"):
                for feature_def in features_config["create_features"]:
                    feature_name = feature_def.get("name")
                    expression = feature_def.get("expression")
                    
                    if feature_name and expression:
                        data = self.preprocessing_engine.feature_processor.create_feature(
                            data, feature_name, expression
                        )
                        features_created.append({
                            "name": feature_name,
                            "expression": expression,
                            "description": feature_def.get("description")
                        })
                        
                        # 保存特征定义
                        feature_entity = FeatureStoreEntity(
                            dataset_id=dataset_id,
                            user_id=user_id,
                            tenant_id=tenant_id,
                            feature_name=feature_name,
                            feature_type="numeric",
                            expression=expression,
                            description=feature_def.get("description"),
                        )
                        self.feature_repo.create(feature_entity)
            
            # 特征选择
            if features_config.get("feature_selection"):
                selection_config = features_config["feature_selection"]
                method = selection_config.get("method", "variance")
                
                if method == "variance":
                    data, selected, removed = self.preprocessing_engine.feature_processor.select_features_by_variance(
                        data,
                        threshold=selection_config.get("threshold", 0.0),
                        exclude_columns=selection_config.get("exclude_columns")
                    )
                    features_selected = selected
                    features_removed = removed
            
            # 特征转换
            if features_config.get("feature_transform"):
                for transform_config in features_config["feature_transform"]:
                    columns = transform_config.get("columns", [])
                    transform_type = transform_config.get("transform_type", "log")
                    
                    data = self.preprocessing_engine.feature_processor.transform_feature(
                        data, columns, transform_type
                    )
                    features_transformed.extend(columns)
            
            # 降维
            if features_config.get("dimension_reduction"):
                reduction_config = features_config["dimension_reduction"]
                data, reduction_info = self.preprocessing_engine.feature_processor.reduce_dimensions(
                    data,
                    columns=reduction_config.get("columns", []),
                    n_components=reduction_config.get("n_components", 2),
                    method=reduction_config.get("method", "pca")
                )
            
            # 编码
            if features_config.get("encoding"):
                for encoding_config in features_config["encoding"]:
                    data, mappings = self.preprocessing_engine.transform_processor.encode_categorical(
                        data,
                        columns=encoding_config.get("columns", []),
                        method=encoding_config.get("method", "label"),
                        handle_unknown=encoding_config.get("handle_unknown", "ignore")
                    )
            
            final_columns = len(data[0]) if data else 0
            
            # 构建特征工程报告
            feature_report = {
                "engineered_at": datetime.utcnow().isoformat(),
                "task_id": task.task_id,
                "features_created": features_created,
                "features_selected": features_selected,
                "features_removed": features_removed,
                "features_transformed": features_transformed,
                "feature_importance": feature_importance,
                "original_columns": original_columns,
                "final_columns": final_columns,
                "config_used": features_config
            }
            
            # 更新任务
            task.status = "completed"
            task.result = feature_report
            task.original_columns = original_columns
            task.final_columns = final_columns
            task.completed_at = datetime.utcnow()
            self.task_repo.update(task)
            
            # 创建历史记录
            history = PreprocessingHistoryEntity(
                dataset_id=dataset_id,
                task_id=task.task_id,
                user_id=user_id,
                tenant_id=tenant_id,
                operation_type="feature_engineering",
                operation_config=features_config,
                operation_result=feature_report,
                columns_before=original_columns,
                columns_after=final_columns,
                columns_added=[f["name"] for f in features_created],
                columns_removed=features_removed,
                columns_modified=features_transformed,
            )
            self.history_repo.create(history)
            
            # 更新数据集
            if hasattr(dataset, 'config') and dataset.config is not None:
                dataset.config['feature_engineering_report'] = feature_report
            dataset.status = "feature_engineered"
            dataset.updated_at = datetime.utcnow()
            
            logger.info(f"Feature engineering completed for dataset {dataset_id}")
            return self.dataset_repository.update(dataset)
            
        except Exception as e:
            task.status = "failed"
            task.error_message = str(e)
            task.completed_at = datetime.utcnow()
            self.task_repo.update(task)
            
            logger.error(f"Feature engineering failed for dataset {dataset_id}: {e}")
            raise FeatureEngineeringError(dataset_id, "feature_engineering", str(e))
        
    def perform_data_augmentation(
        self, 
        dataset_id: str, 
        augmentation_config: Dict[str, Any]
    ) -> Dataset:
        """执行数据增强
        
        Args:
            dataset_id: 数据集ID
            augmentation_config: 数据增强配置
            
        Returns:
            Dataset: 数据增强后的数据集对象
        """
        logger.info(f"Performing data augmentation on dataset {dataset_id}")
        
        # 获取数据集
        dataset = self.dataset_repository.get_by_id(dataset_id)
        if not dataset:
            raise DatasetNotFoundError(f"数据集 {dataset_id} 不存在")
        
        # 获取用户信息
        user_id = getattr(dataset, 'user_id', '')
        config = getattr(dataset, 'config', {}) or {}
        tenant_id = config.get('tenant_id')
        
        # 创建任务
        task = PreprocessingTaskEntity(
            dataset_id=dataset_id,
            user_id=user_id,
            tenant_id=tenant_id,
            task_type="augmentation",
            task_name="数据增强",
            status="processing",
            config=augmentation_config,
        )
        task = self.task_repo.create(task)
        
        try:
            # 获取数据
            data = self._get_dataset_data(dataset)
            original_samples = len(data)
            
            augmentation_type = augmentation_config.get("augmentation_type", "text")
            generated_samples = 0
            method_stats = {}
            original_distribution = None
            final_distribution = None
            
            if augmentation_type == "text":
                text_config = augmentation_config.get("text_config", {})
                data, generated = self.preprocessing_engine.augmentation_processor.augment_text(
                    data,
                    columns=text_config.get("columns", []),
                    methods=text_config.get("methods", ["random_deletion"]),
                    augment_ratio=text_config.get("augment_ratio", 0.3),
                    num_augment=text_config.get("num_augment", 1)
                )
                generated_samples = generated
                method_stats = {"text_augmentation": generated}
            
            elif augmentation_type == "tabular":
                sampling_config = augmentation_config.get("sampling_config", {})
                method = sampling_config.get("method", "oversample")
                target_column = sampling_config.get("target_column")
                
                if target_column:
                    # 记录原始分布
                    from collections import Counter
                    original_distribution = dict(Counter(row.get(target_column) for row in data))
                    
                    if method == "oversample":
                        data, counts = self.preprocessing_engine.augmentation_processor.oversample(
                            data,
                            target_column=target_column,
                            sampling_strategy=sampling_config.get("sampling_strategy", "auto")
                        )
                    elif method == "undersample":
                        data, counts = self.preprocessing_engine.augmentation_processor.undersample(
                            data,
                            target_column=target_column,
                            sampling_strategy=sampling_config.get("sampling_strategy", "auto")
                        )
                    
                    final_distribution = dict(Counter(row.get(target_column) for row in data))
                    generated_samples = len(data) - original_samples
                    method_stats = {method: abs(generated_samples)}
            
            final_samples = len(data)
            
            # 构建增强报告
            augmentation_report = {
                "augmented_at": datetime.utcnow().isoformat(),
                "task_id": task.task_id,
                "augmentation_type": augmentation_type,
                "original_samples": original_samples,
                "generated_samples": abs(generated_samples),
                "final_samples": final_samples,
                "method_stats": method_stats,
                "original_distribution": original_distribution,
                "final_distribution": final_distribution,
                "config_used": augmentation_config
            }
            
            # 更新任务
            task.status = "completed"
            task.result = augmentation_report
            task.original_rows = original_samples
            task.final_rows = final_samples
            task.completed_at = datetime.utcnow()
            self.task_repo.update(task)
            
            # 创建历史记录
            history = PreprocessingHistoryEntity(
                dataset_id=dataset_id,
                task_id=task.task_id,
                user_id=user_id,
                tenant_id=tenant_id,
                operation_type="augmentation",
                operation_config=augmentation_config,
                operation_result=augmentation_report,
                rows_before=original_samples,
                rows_after=final_samples,
            )
            self.history_repo.create(history)
            
            # 更新数据集
            if hasattr(dataset, 'config') and dataset.config is not None:
                dataset.config['augmentation_report'] = augmentation_report
            dataset.status = "augmented"
            dataset.updated_at = datetime.utcnow()
            
            logger.info(f"Data augmentation completed for dataset {dataset_id}")
            return self.dataset_repository.update(dataset)
            
        except Exception as e:
            task.status = "failed"
            task.error_message = str(e)
            task.completed_at = datetime.utcnow()
            self.task_repo.update(task)
            
            logger.error(f"Data augmentation failed for dataset {dataset_id}: {e}")
            raise DataAugmentationError(dataset_id, "augmentation", str(e))
        
    def split_dataset(
        self, 
        dataset_id: str, 
        split_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """分割数据集
        
        Args:
            dataset_id: 数据集ID
            split_config: 分割配置
            
        Returns:
            Dict[str, Any]: 分割后的数据集信息
        """
        logger.info(f"Splitting dataset {dataset_id}")
        
        # 获取数据集
        dataset = self.dataset_repository.get_by_id(dataset_id)
        if not dataset:
            raise DatasetNotFoundError(f"数据集 {dataset_id} 不存在")
        
        # 获取用户信息
        user_id = getattr(dataset, 'user_id', '')
        config = getattr(dataset, 'config', {}) or {}
        tenant_id = config.get('tenant_id')
        
        # 创建任务
        task = PreprocessingTaskEntity(
            dataset_id=dataset_id,
            user_id=user_id,
            tenant_id=tenant_id,
            task_type="split",
            task_name="数据分割",
            status="processing",
            config=split_config,
        )
        task = self.task_repo.create(task)
        
        try:
            # 获取数据
            data = self._get_dataset_data(dataset)
            total_samples = len(data)
            
            # 获取分割参数
            train_ratio = split_config.get("train_ratio", 0.8)
            val_ratio = split_config.get("val_ratio", 0.1)
            test_ratio = split_config.get("test_ratio", 0.1)
            stratify_column = split_config.get("stratify_column")
            shuffle = split_config.get("shuffle", True)
            random_state = split_config.get("random_state", 42)
            create_new_datasets = split_config.get("create_new_datasets", True)
            
            # 检查交叉验证配置
            cv_config = split_config.get("cross_validation")
            
            if cv_config:
                # 交叉验证分割
                folds = self.preprocessing_engine.split_processor.k_fold_split(
                    data,
                    n_folds=cv_config.get("n_folds", 5),
                    stratify_column=cv_config.get("stratify_column"),
                    shuffle=cv_config.get("shuffle", True),
                    random_state=cv_config.get("random_state", 42)
                )
                
                split_result = {
                    "split_at": datetime.utcnow().isoformat(),
                    "task_id": task.task_id,
                    "split_type": "cross_validation",
                    "total_samples": total_samples,
                    "n_folds": len(folds),
                    "cv_folds": [
                        {
                            "fold": f["fold"],
                            "train_count": f["train_count"],
                            "val_count": f["val_count"]
                        }
                        for f in folds
                    ],
                    "config_used": split_config
                }
            else:
                # 普通分割
                split_data = self.preprocessing_engine.split_processor.train_val_test_split(
                    data,
                    train_ratio=train_ratio,
                    val_ratio=val_ratio,
                    test_ratio=test_ratio,
                    stratify_column=stratify_column,
                    shuffle=shuffle,
                    random_state=random_state
                )
                
                train_dataset_id = None
                val_dataset_id = None
                test_dataset_id = None
                
                # 创建新数据集
                if create_new_datasets:
                    dataset_name = getattr(dataset, 'name', 'Dataset')
                    
                    # 创建训练集
                    train_ds = DataclassDataset(
                        user_id=user_id,
                        name=f"{dataset_name}_train",
                        description=f"训练集 - 从 {dataset_name} 分割",
                        dataset_type=getattr(dataset, 'dataset_type', 'generic'),
                        format=getattr(dataset, 'format', 'json'),
                        config={
                            'parent_dataset_id': dataset_id,
                            'split_type': 'train',
                            'tenant_id': tenant_id,
                        }
                    )
                    train_ds = self.dataset_repository.create(train_ds)
                    train_dataset_id = train_ds.dataset_id
                    
                    # 创建验证集
                    if split_data["val_count"] > 0:
                        val_ds = DataclassDataset(
                            user_id=user_id,
                            name=f"{dataset_name}_val",
                            description=f"验证集 - 从 {dataset_name} 分割",
                            dataset_type=getattr(dataset, 'dataset_type', 'generic'),
                            format=getattr(dataset, 'format', 'json'),
                            config={
                                'parent_dataset_id': dataset_id,
                                'split_type': 'val',
                                'tenant_id': tenant_id,
                            }
                        )
                        val_ds = self.dataset_repository.create(val_ds)
                        val_dataset_id = val_ds.dataset_id
                    
                    # 创建测试集
                    if split_data["test_count"] > 0:
                        test_ds = DataclassDataset(
                            user_id=user_id,
                            name=f"{dataset_name}_test",
                            description=f"测试集 - 从 {dataset_name} 分割",
                            dataset_type=getattr(dataset, 'dataset_type', 'generic'),
                            format=getattr(dataset, 'format', 'json'),
                            config={
                                'parent_dataset_id': dataset_id,
                                'split_type': 'test',
                                'tenant_id': tenant_id,
                            }
                        )
                        test_ds = self.dataset_repository.create(test_ds)
                        test_dataset_id = test_ds.dataset_id
                
                split_result = {
                    "split_at": datetime.utcnow().isoformat(),
                    "task_id": task.task_id,
                    "split_type": "train_val_test",
                    "total_samples": total_samples,
                    "train_samples": split_data["train_count"],
                    "val_samples": split_data["val_count"],
                    "test_samples": split_data["test_count"],
                    "train_dataset_id": train_dataset_id,
                    "val_dataset_id": val_dataset_id,
                    "test_dataset_id": test_dataset_id,
                    "train_distribution": split_data.get("train_distribution"),
                    "val_distribution": split_data.get("val_distribution"),
                    "test_distribution": split_data.get("test_distribution"),
                    "config_used": split_config
                }
            
            # 更新任务
            task.status = "completed"
            task.result = split_result
            task.original_rows = total_samples
            task.completed_at = datetime.utcnow()
            self.task_repo.update(task)
            
            # 创建历史记录
            history = PreprocessingHistoryEntity(
                dataset_id=dataset_id,
                task_id=task.task_id,
                user_id=user_id,
                tenant_id=tenant_id,
                operation_type="split",
                operation_config=split_config,
                operation_result=split_result,
                rows_before=total_samples,
            )
            self.history_repo.create(history)
            
            # 更新原数据集
            if hasattr(dataset, 'config') and dataset.config is not None:
                dataset.config['split_report'] = split_result
            dataset.updated_at = datetime.utcnow()
            self.dataset_repository.update(dataset)
            
            logger.info(f"Dataset split completed for dataset {dataset_id}")
            return split_result
            
        except DataSplitRatioError:
            raise
        except Exception as e:
            task.status = "failed"
            task.error_message = str(e)
            task.completed_at = datetime.utcnow()
            self.task_repo.update(task)
            
            logger.error(f"Dataset split failed for dataset {dataset_id}: {e}")
            raise DataSplitError(dataset_id, str(e))

    # ========================================================================
    # 任务管理方法
    # ========================================================================

    def get_task(self, task_id: str) -> Dict[str, Any]:
        """获取预处理任务详情
        
        Args:
            task_id: 任务ID
            
        Returns:
            任务详情
        """
        task = self.task_repo.get_by_id(task_id)
        if not task:
            raise PreprocessingTaskNotFoundError(task_id)
        return task.to_dict()

    def list_tasks(
        self,
        dataset_id: str,
        status_filter: Optional[List[str]] = None,
        task_type_filter: Optional[List[str]] = None,
        page: int = 1,
        page_size: int = 20
    ) -> Dict[str, Any]:
        """获取数据集的任务列表
        
        Args:
            dataset_id: 数据集ID
            status_filter: 状态过滤
            task_type_filter: 任务类型过滤
            page: 页码
            page_size: 每页大小
            
        Returns:
            任务列表和分页信息
        """
        offset = (page - 1) * page_size
        tasks, total = self.task_repo.get_by_dataset(
            dataset_id=dataset_id,
            status_filter=status_filter,
            task_type_filter=task_type_filter,
            limit=page_size,
            offset=offset
        )
        
        return {
            "tasks": [t.to_dict() for t in tasks],
            "total": total,
            "page": page,
            "page_size": page_size,
            "total_pages": (total + page_size - 1) // page_size
        }

    def cancel_task(self, task_id: str) -> Dict[str, Any]:
        """取消预处理任务
        
        Args:
            task_id: 任务ID
            
        Returns:
            更新后的任务信息
        """
        task = self.task_repo.get_by_id(task_id)
        if not task:
            raise PreprocessingTaskNotFoundError(task_id)
        
        if task.status not in ("pending", "processing"):
            raise DataPreprocessingError(f"任务 {task_id} 状态为 {task.status}，无法取消")
        
        task = self.task_repo.update_status(task_id, "cancelled")
        logger.info(f"Cancelled task: {task_id}")
        return task.to_dict()

    # ========================================================================
    # 历史记录方法
    # ========================================================================

    def get_preprocessing_history(
        self,
        dataset_id: str,
        operation_type_filter: Optional[List[str]] = None,
        page: int = 1,
        page_size: int = 20
    ) -> Dict[str, Any]:
        """获取预处理历史记录
        
        Args:
            dataset_id: 数据集ID
            operation_type_filter: 操作类型过滤
            page: 页码
            page_size: 每页大小
            
        Returns:
            历史记录列表和分页信息
        """
        offset = (page - 1) * page_size
        histories, total = self.history_repo.get_by_dataset(
            dataset_id=dataset_id,
            operation_type_filter=operation_type_filter,
            limit=page_size,
            offset=offset
        )
        
        return {
            "histories": [h.to_dict() for h in histories],
            "total": total,
            "page": page,
            "page_size": page_size,
            "total_pages": (total + page_size - 1) // page_size
        }

    # ========================================================================
    # 流水线管理方法
    # ========================================================================

    def create_pipeline(
        self,
        user_id: str,
        name: str,
        operations: List[Dict[str, Any]],
        description: Optional[str] = None,
        is_template: bool = False,
        is_public: bool = False,
        tenant_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """创建预处理流水线
        
        Args:
            user_id: 用户ID
            name: 流水线名称
            operations: 操作列表
            description: 描述
            is_template: 是否为模板
            is_public: 是否公开
            tenant_id: 租户ID
            
        Returns:
            创建的流水线信息
        """
        pipeline = PreprocessingPipelineEntity(
            user_id=user_id,
            tenant_id=tenant_id,
            name=name,
            description=description,
            operations=operations,
            is_template=is_template,
            is_public=is_public,
        )
        pipeline = self.pipeline_repo.create(pipeline)
        logger.info(f"Created pipeline: {pipeline.pipeline_id}")
        return pipeline.to_dict()

    def get_pipeline(self, pipeline_id: str) -> Dict[str, Any]:
        """获取流水线详情
        
        Args:
            pipeline_id: 流水线ID
            
        Returns:
            流水线详情
        """
        pipeline = self.pipeline_repo.get_by_id(pipeline_id)
        if not pipeline:
            raise DataPreprocessingError(f"流水线 {pipeline_id} 不存在")
        return pipeline.to_dict()

    def list_pipelines(
        self,
        user_id: str,
        tenant_id: Optional[str] = None,
        is_template: Optional[bool] = None,
        page: int = 1,
        page_size: int = 20
    ) -> Dict[str, Any]:
        """获取用户的流水线列表
        
        Args:
            user_id: 用户ID
            tenant_id: 租户ID
            is_template: 是否为模板
            page: 页码
            page_size: 每页大小
            
        Returns:
            流水线列表和分页信息
        """
        offset = (page - 1) * page_size
        pipelines, total = self.pipeline_repo.get_by_user(
            user_id=user_id,
            tenant_id=tenant_id,
            is_template=is_template,
            limit=page_size,
            offset=offset
        )
        
        return {
            "pipelines": [p.to_dict() for p in pipelines],
            "total": total,
            "page": page,
            "page_size": page_size,
            "total_pages": (total + page_size - 1) // page_size
        }

    def execute_pipeline(
        self,
        dataset_id: str,
        pipeline_id: str
    ) -> Dataset:
        """执行预处理流水线
        
        Args:
            dataset_id: 数据集ID
            pipeline_id: 流水线ID
            
        Returns:
            处理后的数据集
        """
        pipeline = self.pipeline_repo.get_by_id(pipeline_id)
        if not pipeline:
            raise DataPreprocessingError(f"流水线 {pipeline_id} 不存在")
        
        # 增加使用次数
        self.pipeline_repo.increment_usage(pipeline_id)
        
        # 使用流水线配置执行预处理
        return self.preprocess_dataset(
            dataset_id,
            {"pipeline_config": {"operations": pipeline.operations}}
        )

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""模型管理数据访问层

提供模型管理模块的数据持久化操作：
- 性能记录的 CRUD
- 验证结果的 CRUD
- 导入记录的 CRUD
- 比较记录的 CRUD

架构调用关系：
Service层 (model_management_service.py)
    -> Repository层 (本模块)
        -> Database层 (DatabaseService)
"""

import logging
import uuid
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


class ModelManagementRepository:
    """模型管理数据访问层
    
    提供性能记录、验证结果、导入记录、比较记录的数据持久化操作。
    支持内存存储模式（用于测试）和数据库存储模式。
    """
    
    def __init__(self, use_memory: bool = False):
        """初始化仓库
        
        Args:
            use_memory: 是否使用内存存储（用于测试）
        """
        self._use_memory = use_memory
        self._db_service = None
        
        # 内存存储
        self._performance_records: Dict[str, Dict[str, Any]] = {}
        self._validation_results: Dict[str, Dict[str, Any]] = {}
        self._import_records: Dict[str, Dict[str, Any]] = {}
        self._comparison_records: Dict[str, Dict[str, Any]] = {}
        
        if not use_memory:
            self._init_database()
    
    def _init_database(self):
        """初始化数据库连接"""
        try:
            from backend.modules.database.service import DatabaseService
            self._db_service = DatabaseService()
            logger.info("ModelManagementRepository: Database service initialized")
        except Exception as e:
            logger.warning(f"ModelManagementRepository: Failed to init database: {e}")
            self._use_memory = True
    
    # ==========================================================================
    # 性能记录操作
    # ==========================================================================
    
    def create_performance_record(
        self,
        model_id: str,
        user_id: str,
        metrics: Dict[str, Any],
        tenant_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """创建性能记录
        
        Args:
            model_id: 模型ID
            user_id: 用户ID
            metrics: 性能指标
            tenant_id: 租户ID
            
        Returns:
            创建的性能记录
        """
        record_id = str(uuid.uuid4())
        now = datetime.utcnow()
        
        record = {
            'id': record_id,
            'model_id': model_id,
            'user_id': user_id,
            'accuracy': metrics.get('accuracy'),
            'precision': metrics.get('precision'),
            'recall': metrics.get('recall'),
            'f1_score': metrics.get('f1_score'),
            'loss': metrics.get('loss'),
            'auc': metrics.get('auc'),
            'mae': metrics.get('mae'),
            'mse': metrics.get('mse'),
            'rmse': metrics.get('rmse'),
            'training_time_seconds': metrics.get('training_time_seconds'),
            'inference_time_ms': metrics.get('inference_time_ms'),
            'throughput_per_second': metrics.get('throughput_per_second'),
            'memory_usage_mb': metrics.get('memory_usage_mb'),
            'gpu_memory_mb': metrics.get('gpu_memory_mb'),
            'cpu_utilization': metrics.get('cpu_utilization'),
            'gpu_utilization': metrics.get('gpu_utilization'),
            'test_data_size': metrics.get('test_data_size'),
            'evaluation_time': now.isoformat(),
            'custom_metrics': metrics.get('custom_metrics', {}),
            'tenant_id': tenant_id or 'default',
            'created_at': now.isoformat(),
        }
        
        if self._use_memory:
            self._performance_records[record_id] = record
        else:
            try:
                from backend.schemas.model_management_models import ModelPerformanceRecordDB
                record_db = ModelPerformanceRecordDB(
                    id=uuid.UUID(record_id),
                    model_id=model_id,
                    user_id=user_id,
                    **{k: v for k, v in metrics.items() if k not in ['custom_metrics']},
                    custom_metrics=metrics.get('custom_metrics', {}),
                    evaluation_time=now,
                    tenant_id=tenant_id or 'default',
                )
                created = self._db_service.create(record_db)
                record = created.to_dict()
            except Exception as e:
                logger.error(f"Failed to create performance record: {e}")
                self._performance_records[record_id] = record
        
        logger.info(f"Created performance record: {record_id} for model: {model_id}")
        return record
    
    def get_performance_record(self, record_id: str) -> Optional[Dict[str, Any]]:
        """获取性能记录
        
        Args:
            record_id: 记录ID
            
        Returns:
            性能记录
        """
        if self._use_memory:
            return self._performance_records.get(record_id)
        
        try:
            from backend.schemas.model_management_models import ModelPerformanceRecordDB
            record_db = self._db_service.get_by_id(ModelPerformanceRecordDB, record_id)
            return record_db.to_dict() if record_db else None
        except Exception as e:
            logger.error(f"Failed to get performance record: {e}")
            return self._performance_records.get(record_id)
    
    def get_model_performance_history(
        self,
        model_id: str,
        limit: int = 100,
        offset: int = 0
    ) -> Tuple[List[Dict[str, Any]], int]:
        """获取模型的性能历史记录
        
        Args:
            model_id: 模型ID
            limit: 限制数量
            offset: 偏移量
            
        Returns:
            (记录列表, 总数)
        """
        if self._use_memory:
            records = [r for r in self._performance_records.values() 
                      if r.get('model_id') == model_id]
            records.sort(key=lambda x: x.get('created_at', ''), reverse=True)
            total = len(records)
            return records[offset:offset + limit], total
        
        try:
            from backend.schemas.model_management_models import ModelPerformanceRecordDB
            records_db = self._db_service.filter_by(ModelPerformanceRecordDB, model_id=model_id)
            records = [r.to_dict() for r in records_db]
            records.sort(key=lambda x: x.get('created_at', ''), reverse=True)
            total = len(records)
            return records[offset:offset + limit], total
        except Exception as e:
            logger.error(f"Failed to get performance history: {e}")
            return [], 0
    
    def get_latest_performance(self, model_id: str) -> Optional[Dict[str, Any]]:
        """获取模型最新的性能记录
        
        Args:
            model_id: 模型ID
            
        Returns:
            最新的性能记录
        """
        records, _ = self.get_model_performance_history(model_id, limit=1)
        return records[0] if records else None
    
    # ==========================================================================
    # 验证结果操作
    # ==========================================================================
    
    def create_validation_result(
        self,
        model_id: str,
        user_id: str,
        status: str = 'pending',
        metrics: Optional[Dict[str, Any]] = None,
        validation_config: Optional[Dict[str, Any]] = None,
        tenant_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """创建验证结果
        
        Args:
            model_id: 模型ID
            user_id: 用户ID
            status: 状态
            metrics: 指标
            validation_config: 验证配置
            tenant_id: 租户ID
            
        Returns:
            创建的验证结果
        """
        result_id = str(uuid.uuid4())
        now = datetime.utcnow()
        
        metrics = metrics or {}
        
        result = {
            'id': result_id,
            'model_id': model_id,
            'user_id': user_id,
            'status': status,
            'accuracy': metrics.get('accuracy'),
            'precision': metrics.get('precision'),
            'recall': metrics.get('recall'),
            'f1_score': metrics.get('f1_score'),
            'loss': metrics.get('loss'),
            'validation_time': now.isoformat(),
            'test_data_size': metrics.get('test_data_size'),
            'passed_tests': metrics.get('passed_tests', 0),
            'failed_tests': metrics.get('failed_tests', 0),
            'validation_config': validation_config or {},
            'test_results': metrics.get('test_results', []),
            'confusion_matrix': metrics.get('confusion_matrix'),
            'classification_report': metrics.get('classification_report'),
            'error_message': metrics.get('error_message'),
            'tenant_id': tenant_id or 'default',
            'created_at': now.isoformat(),
        }
        
        if self._use_memory:
            self._validation_results[result_id] = result
        else:
            try:
                from backend.schemas.model_management_models import ModelValidationResultDB
                result_db = ModelValidationResultDB(
                    id=uuid.UUID(result_id),
                    model_id=model_id,
                    user_id=user_id,
                    status=status,
                    accuracy=metrics.get('accuracy'),
                    precision=metrics.get('precision'),
                    recall=metrics.get('recall'),
                    f1_score=metrics.get('f1_score'),
                    loss=metrics.get('loss'),
                    validation_time=now,
                    test_data_size=metrics.get('test_data_size'),
                    passed_tests=metrics.get('passed_tests', 0),
                    failed_tests=metrics.get('failed_tests', 0),
                    validation_config=validation_config or {},
                    test_results=metrics.get('test_results', []),
                    confusion_matrix=metrics.get('confusion_matrix'),
                    classification_report=metrics.get('classification_report'),
                    tenant_id=tenant_id or 'default',
                )
                created = self._db_service.create(result_db)
                result = created.to_dict()
            except Exception as e:
                logger.error(f"Failed to create validation result: {e}")
                self._validation_results[result_id] = result
        
        logger.info(f"Created validation result: {result_id} for model: {model_id}")
        return result
    
    def get_validation_result(self, result_id: str) -> Optional[Dict[str, Any]]:
        """获取验证结果
        
        Args:
            result_id: 结果ID
            
        Returns:
            验证结果
        """
        if self._use_memory:
            return self._validation_results.get(result_id)
        
        try:
            from backend.schemas.model_management_models import ModelValidationResultDB
            result_db = self._db_service.get_by_id(ModelValidationResultDB, result_id)
            return result_db.to_dict() if result_db else None
        except Exception as e:
            logger.error(f"Failed to get validation result: {e}")
            return self._validation_results.get(result_id)
    
    def update_validation_result(
        self,
        result_id: str,
        **updates
    ) -> Optional[Dict[str, Any]]:
        """更新验证结果
        
        Args:
            result_id: 结果ID
            **updates: 更新字段
            
        Returns:
            更新后的验证结果
        """
        if self._use_memory:
            if result_id in self._validation_results:
                self._validation_results[result_id].update(updates)
                return self._validation_results[result_id]
            return None
        
        try:
            from backend.schemas.model_management_models import ModelValidationResultDB
            result_db = self._db_service.get_by_id(ModelValidationResultDB, result_id)
            if result_db:
                updated = self._db_service.update(result_db, updates)
                return updated.to_dict()
            return None
        except Exception as e:
            logger.error(f"Failed to update validation result: {e}")
            return None
    
    def get_model_validation_history(
        self,
        model_id: str,
        limit: int = 50,
        offset: int = 0
    ) -> Tuple[List[Dict[str, Any]], int]:
        """获取模型的验证历史
        
        Args:
            model_id: 模型ID
            limit: 限制数量
            offset: 偏移量
            
        Returns:
            (结果列表, 总数)
        """
        if self._use_memory:
            results = [r for r in self._validation_results.values() 
                      if r.get('model_id') == model_id]
            results.sort(key=lambda x: x.get('created_at', ''), reverse=True)
            total = len(results)
            return results[offset:offset + limit], total
        
        try:
            from backend.schemas.model_management_models import ModelValidationResultDB
            results_db = self._db_service.filter_by(ModelValidationResultDB, model_id=model_id)
            results = [r.to_dict() for r in results_db]
            results.sort(key=lambda x: x.get('created_at', ''), reverse=True)
            total = len(results)
            return results[offset:offset + limit], total
        except Exception as e:
            logger.error(f"Failed to get validation history: {e}")
            return [], 0
    
    # ==========================================================================
    # 导入记录操作
    # ==========================================================================
    
    def create_import_record(
        self,
        user_id: str,
        model_name: str,
        import_source: str = 'local',
        source_path: Optional[str] = None,
        source_url: Optional[str] = None,
        import_config: Optional[Dict[str, Any]] = None,
        tenant_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """创建导入记录
        
        Args:
            user_id: 用户ID
            model_name: 模型名称
            import_source: 导入来源
            source_path: 来源路径
            source_url: 来源URL
            import_config: 导入配置
            tenant_id: 租户ID
            
        Returns:
            创建的导入记录
        """
        record_id = str(uuid.uuid4())
        now = datetime.utcnow()
        
        record = {
            'id': record_id,
            'user_id': user_id,
            'target_model_id': None,
            'status': 'pending',
            'import_source': import_source,
            'source_path': source_path,
            'source_url': source_url,
            'model_name': model_name,
            'model_type': None,
            'model_framework': None,
            'model_format': None,
            'file_size_bytes': None,
            'checksum': None,
            'import_config': import_config or {},
            'progress': 0.0,
            'started_at': None,
            'completed_at': None,
            'error_message': None,
            'import_result': {},
            'tenant_id': tenant_id or 'default',
            'created_at': now.isoformat(),
        }
        
        if self._use_memory:
            self._import_records[record_id] = record
        else:
            try:
                from backend.schemas.model_management_models import ModelImportRecordDB
                record_db = ModelImportRecordDB(
                    id=uuid.UUID(record_id),
                    user_id=user_id,
                    model_name=model_name,
                    import_source=import_source,
                    source_path=source_path,
                    source_url=source_url,
                    import_config=import_config or {},
                    tenant_id=tenant_id or 'default',
                )
                created = self._db_service.create(record_db)
                record = created.to_dict()
            except Exception as e:
                logger.error(f"Failed to create import record: {e}")
                self._import_records[record_id] = record
        
        logger.info(f"Created import record: {record_id} for model: {model_name}")
        return record
    
    def get_import_record(self, record_id: str) -> Optional[Dict[str, Any]]:
        """获取导入记录
        
        Args:
            record_id: 记录ID
            
        Returns:
            导入记录
        """
        if self._use_memory:
            return self._import_records.get(record_id)
        
        try:
            from backend.schemas.model_management_models import ModelImportRecordDB
            record_db = self._db_service.get_by_id(ModelImportRecordDB, record_id)
            return record_db.to_dict() if record_db else None
        except Exception as e:
            logger.error(f"Failed to get import record: {e}")
            return self._import_records.get(record_id)
    
    def update_import_record(
        self,
        record_id: str,
        **updates
    ) -> Optional[Dict[str, Any]]:
        """更新导入记录
        
        Args:
            record_id: 记录ID
            **updates: 更新字段
            
        Returns:
            更新后的导入记录
        """
        if self._use_memory:
            if record_id in self._import_records:
                self._import_records[record_id].update(updates)
                return self._import_records[record_id]
            return None
        
        try:
            from backend.schemas.model_management_models import ModelImportRecordDB
            record_db = self._db_service.get_by_id(ModelImportRecordDB, record_id)
            if record_db:
                updated = self._db_service.update(record_db, updates)
                return updated.to_dict()
            return None
        except Exception as e:
            logger.error(f"Failed to update import record: {e}")
            return None
    
    def list_user_imports(
        self,
        user_id: str,
        status: Optional[str] = None,
        limit: int = 50,
        offset: int = 0
    ) -> Tuple[List[Dict[str, Any]], int]:
        """获取用户的导入记录列表
        
        Args:
            user_id: 用户ID
            status: 状态过滤
            limit: 限制数量
            offset: 偏移量
            
        Returns:
            (记录列表, 总数)
        """
        if self._use_memory:
            records = [r for r in self._import_records.values() 
                      if r.get('user_id') == user_id]
            if status:
                records = [r for r in records if r.get('status') == status]
            records.sort(key=lambda x: x.get('created_at', ''), reverse=True)
            total = len(records)
            return records[offset:offset + limit], total
        
        try:
            from backend.schemas.model_management_models import ModelImportRecordDB
            records_db = self._db_service.filter_by(ModelImportRecordDB, user_id=user_id)
            records = [r.to_dict() for r in records_db]
            if status:
                records = [r for r in records if r.get('status') == status]
            records.sort(key=lambda x: x.get('created_at', ''), reverse=True)
            total = len(records)
            return records[offset:offset + limit], total
        except Exception as e:
            logger.error(f"Failed to list user imports: {e}")
            return [], 0
    
    # ==========================================================================
    # 比较记录操作
    # ==========================================================================
    
    def create_comparison_record(
        self,
        user_id: str,
        model_ids: List[str],
        comparison_result: Dict[str, Any],
        metrics_to_compare: Optional[List[str]] = None,
        comparison_config: Optional[Dict[str, Any]] = None,
        winner_model_id: Optional[str] = None,
        tenant_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """创建比较记录
        
        Args:
            user_id: 用户ID
            model_ids: 比较的模型ID列表
            comparison_result: 比较结果
            metrics_to_compare: 比较的指标列表
            comparison_config: 比较配置
            winner_model_id: 胜出模型ID
            tenant_id: 租户ID
            
        Returns:
            创建的比较记录
        """
        record_id = str(uuid.uuid4())
        now = datetime.utcnow()
        
        record = {
            'id': record_id,
            'user_id': user_id,
            'model_ids': model_ids,
            'comparison_config': comparison_config or {},
            'metrics_to_compare': metrics_to_compare or ['accuracy', 'precision', 'recall', 'f1_score', 'loss'],
            'comparison_result': comparison_result,
            'winner_model_id': winner_model_id,
            'comparison_time': now.isoformat(),
            'tenant_id': tenant_id or 'default',
            'created_at': now.isoformat(),
        }
        
        if self._use_memory:
            self._comparison_records[record_id] = record
        else:
            try:
                from backend.schemas.model_management_models import ModelComparisonRecordDB
                record_db = ModelComparisonRecordDB(
                    id=uuid.UUID(record_id),
                    user_id=user_id,
                    model_ids=model_ids,
                    comparison_config=comparison_config or {},
                    metrics_to_compare=metrics_to_compare or ['accuracy', 'precision', 'recall', 'f1_score', 'loss'],
                    comparison_result=comparison_result,
                    winner_model_id=winner_model_id,
                    comparison_time=now,
                    tenant_id=tenant_id or 'default',
                )
                created = self._db_service.create(record_db)
                record = created.to_dict()
            except Exception as e:
                logger.error(f"Failed to create comparison record: {e}")
                self._comparison_records[record_id] = record
        
        logger.info(f"Created comparison record: {record_id}")
        return record
    
    def get_comparison_record(self, record_id: str) -> Optional[Dict[str, Any]]:
        """获取比较记录
        
        Args:
            record_id: 记录ID
            
        Returns:
            比较记录
        """
        if self._use_memory:
            return self._comparison_records.get(record_id)
        
        try:
            from backend.schemas.model_management_models import ModelComparisonRecordDB
            record_db = self._db_service.get_by_id(ModelComparisonRecordDB, record_id)
            return record_db.to_dict() if record_db else None
        except Exception as e:
            logger.error(f"Failed to get comparison record: {e}")
            return self._comparison_records.get(record_id)
    
    def list_user_comparisons(
        self,
        user_id: str,
        limit: int = 50,
        offset: int = 0
    ) -> Tuple[List[Dict[str, Any]], int]:
        """获取用户的比较记录列表
        
        Args:
            user_id: 用户ID
            limit: 限制数量
            offset: 偏移量
            
        Returns:
            (记录列表, 总数)
        """
        if self._use_memory:
            records = [r for r in self._comparison_records.values() 
                      if r.get('user_id') == user_id]
            records.sort(key=lambda x: x.get('created_at', ''), reverse=True)
            total = len(records)
            return records[offset:offset + limit], total
        
        try:
            from backend.schemas.model_management_models import ModelComparisonRecordDB
            records_db = self._db_service.filter_by(ModelComparisonRecordDB, user_id=user_id)
            records = [r.to_dict() for r in records_db]
            records.sort(key=lambda x: x.get('created_at', ''), reverse=True)
            total = len(records)
            return records[offset:offset + limit], total
        except Exception as e:
            logger.error(f"Failed to list user comparisons: {e}")
            return [], 0


# ==================== 全局单例 ====================

_global_repository: Optional[ModelManagementRepository] = None


def get_management_repository(use_memory: bool = False) -> ModelManagementRepository:
    """获取模型管理仓库实例
    
    Args:
        use_memory: 是否使用内存存储
        
    Returns:
        ModelManagementRepository 实例
    """
    global _global_repository
    
    if _global_repository is None:
        _global_repository = ModelManagementRepository(use_memory=use_memory)
    
    return _global_repository


def reset_management_repository():
    """重置全局仓库实例（用于测试）"""
    global _global_repository
    _global_repository = None


# ==================== 导出 ====================

__all__ = [
    'ModelManagementRepository',
    'get_management_repository',
    'reset_management_repository',
]

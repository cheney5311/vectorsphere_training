"""模型评估数据访问层

提供模型评估相关的数据库访问功能，包括：
- 模型评估记录 (ModelEvaluation)
- 评估指标 (ModelEvaluationMetric)
- 模型对比记录 (ModelComparisonRecord)
"""

import json
import logging
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime
import uuid

from backend.core.exceptions import ValidationError, DatabaseError

logger = logging.getLogger(__name__)


def _generate_id() -> str:
    """生成唯一ID"""
    return str(uuid.uuid4())


class ModelEvaluationRepository:
    """模型评估数据访问层"""
    
    def __init__(self, use_memory_storage: bool = False):
        """初始化模型评估仓库
        
        Args:
            use_memory_storage: 是否使用内存存储
        """
        self._use_memory_storage = use_memory_storage
        
        if use_memory_storage:
            self._evaluations: Dict[str, Dict] = {}
            self._metrics: Dict[str, Dict] = {}
            self._db_manager = None
        else:
            try:
                from backend.modules.database.manager import get_database_manager
                self._db_manager = get_database_manager()
            except ImportError:
                logger.warning("Cannot import database service, falling back to memory storage")
                self._use_memory_storage = True
                self._evaluations: Dict[str, Dict] = {}
                self._metrics: Dict[str, Dict] = {}
                self._db_manager = None
    
    def create(self, evaluation_data: Dict[str, Any]) -> Dict[str, Any]:
        """创建评估记录
        
        Args:
            evaluation_data: 评估数据
            
        Returns:
            创建的评估记录
        """
        try:
            record_id = evaluation_data.get('id') or _generate_id()
            evaluation_id = evaluation_data.get('evaluation_id') or f"eval_{_generate_id()[:8]}"
            
            if self._use_memory_storage:
                evaluation_data['id'] = record_id
                evaluation_data['evaluation_id'] = evaluation_id
                evaluation_data['created_at'] = datetime.utcnow().isoformat()
                evaluation_data['updated_at'] = datetime.utcnow().isoformat()
                evaluation_data.setdefault('status', 'pending')
                self._evaluations[record_id] = evaluation_data
                return evaluation_data
            
            from backend.schemas.training_models import ModelEvaluation
            
            with self._db_manager.get_db_session() as db:
                # 处理 JSON 字段
                evaluation_config = evaluation_data.get('evaluation_config', {})
                if isinstance(evaluation_config, dict):
                    evaluation_config = json.dumps(evaluation_config)
                
                metrics_summary = evaluation_data.get('metrics_summary', {})
                if isinstance(metrics_summary, dict):
                    metrics_summary = json.dumps(metrics_summary)
                
                data_statistics = evaluation_data.get('data_statistics', {})
                if isinstance(data_statistics, dict):
                    data_statistics = json.dumps(data_statistics)
                
                tags = evaluation_data.get('tags', [])
                if isinstance(tags, list):
                    tags = json.dumps(tags)
                
                metadata = evaluation_data.get('metadata', {}) or evaluation_data.get('metadata_', {})
                if isinstance(metadata, dict):
                    metadata = json.dumps(metadata)
                
                error_details = evaluation_data.get('error_details', {})
                if isinstance(error_details, dict):
                    error_details = json.dumps(error_details)
                
                evaluation = ModelEvaluation(
                    id=record_id,
                    tenant_id=evaluation_data.get('tenant_id'),
                    evaluation_id=evaluation_id,
                    model_id=evaluation_data['model_id'],
                    dataset_id=evaluation_data['dataset_id'],
                    user_id=evaluation_data.get('user_id'),
                    evaluation_type=evaluation_data.get('evaluation_type', 'automated'),
                    validation_strategy=evaluation_data.get('validation_strategy', 'holdout'),
                    cross_validation_folds=evaluation_data.get('cross_validation_folds', 5),
                    test_size=evaluation_data.get('test_size', 0.2),
                    status=evaluation_data.get('status', 'pending'),
                    accuracy=evaluation_data.get('accuracy'),
                    precision=evaluation_data.get('precision'),
                    recall=evaluation_data.get('recall'),
                    f1_score=evaluation_data.get('f1_score'),
                    auc=evaluation_data.get('auc'),
                    loss=evaluation_data.get('loss'),
                    started_at=evaluation_data.get('started_at'),
                    completed_at=evaluation_data.get('completed_at'),
                    duration_seconds=evaluation_data.get('duration_seconds'),
                    evaluation_config=evaluation_config,
                    metrics_summary=metrics_summary,
                    data_statistics=data_statistics,
                    error_message=evaluation_data.get('error_message'),
                    error_details=error_details,
                    tags=tags,
                    metadata_=metadata
                )
                db.add(evaluation)
                db.commit()
                db.refresh(evaluation)
                return evaluation.to_dict()
                
        except Exception as e:
            logger.error(f"Failed to create evaluation: {e}")
            raise DatabaseError(f"Failed to create evaluation: {e}", operation="create_evaluation")
    
    def get_by_id(self, evaluation_id: str) -> Optional[Dict[str, Any]]:
        """根据评估ID获取记录"""
        try:
            if self._use_memory_storage:
                for eval_data in self._evaluations.values():
                    if eval_data.get('evaluation_id') == evaluation_id:
                        return eval_data
                return None
            
            from backend.schemas.training_models import ModelEvaluation
            
            with self._db_manager.get_db_session() as db:
                evaluation = db.query(ModelEvaluation).filter(
                    ModelEvaluation.evaluation_id == evaluation_id
                ).first()
                return evaluation.to_dict() if evaluation else None
                
        except Exception as e:
            logger.error(f"Failed to get evaluation: {e}")
            return None
    
    def list_by_model(
        self,
        model_id: str,
        tenant_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> Tuple[List[Dict[str, Any]], int]:
        """获取模型的评估记录列表"""
        try:
            if self._use_memory_storage:
                evaluations = [
                    e for e in self._evaluations.values()
                    if e.get('model_id') == model_id
                ]
                if tenant_id:
                    evaluations = [e for e in evaluations if e.get('tenant_id') == tenant_id]
                if status:
                    evaluations = [e for e in evaluations if e.get('status') == status]
                
                evaluations.sort(key=lambda x: x.get('created_at', ''), reverse=True)
                total = len(evaluations)
                return evaluations[offset:offset + limit], total
            
            from backend.schemas.training_models import ModelEvaluation
            
            with self._db_manager.get_db_session() as db:
                query = db.query(ModelEvaluation).filter(
                    ModelEvaluation.model_id == model_id
                )
                if tenant_id:
                    query = query.filter(ModelEvaluation.tenant_id == tenant_id)
                if status:
                    query = query.filter(ModelEvaluation.status == status)
                
                total = query.count()
                evaluations = query.order_by(
                    ModelEvaluation.created_at.desc()
                ).offset(offset).limit(limit).all()
                
                return [e.to_dict() for e in evaluations], total
                
        except Exception as e:
            logger.error(f"Failed to list evaluations by model: {e}")
            return [], 0
    
    def list_by_dataset(
        self,
        dataset_id: str,
        tenant_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> Tuple[List[Dict[str, Any]], int]:
        """获取数据集的评估记录列表"""
        try:
            if self._use_memory_storage:
                evaluations = [
                    e for e in self._evaluations.values()
                    if e.get('dataset_id') == dataset_id
                ]
                if tenant_id:
                    evaluations = [e for e in evaluations if e.get('tenant_id') == tenant_id]
                
                evaluations.sort(key=lambda x: x.get('created_at', ''), reverse=True)
                total = len(evaluations)
                return evaluations[offset:offset + limit], total
            
            from backend.schemas.training_models import ModelEvaluation
            
            with self._db_manager.get_db_session() as db:
                query = db.query(ModelEvaluation).filter(
                    ModelEvaluation.dataset_id == dataset_id
                )
                if tenant_id:
                    query = query.filter(ModelEvaluation.tenant_id == tenant_id)
                
                total = query.count()
                evaluations = query.order_by(
                    ModelEvaluation.created_at.desc()
                ).offset(offset).limit(limit).all()
                
                return [e.to_dict() for e in evaluations], total
                
        except Exception as e:
            logger.error(f"Failed to list evaluations by dataset: {e}")
            return [], 0
    
    def list_by_user(
        self,
        user_id: str,
        tenant_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> Tuple[List[Dict[str, Any]], int]:
        """获取用户的评估记录列表"""
        try:
            if self._use_memory_storage:
                evaluations = [
                    e for e in self._evaluations.values()
                    if e.get('user_id') == user_id
                ]
                if tenant_id:
                    evaluations = [e for e in evaluations if e.get('tenant_id') == tenant_id]
                if status:
                    evaluations = [e for e in evaluations if e.get('status') == status]
                
                evaluations.sort(key=lambda x: x.get('created_at', ''), reverse=True)
                total = len(evaluations)
                return evaluations[offset:offset + limit], total
            
            from backend.schemas.training_models import ModelEvaluation
            
            with self._db_manager.get_db_session() as db:
                query = db.query(ModelEvaluation).filter(
                    ModelEvaluation.user_id == user_id
                )
                if tenant_id:
                    query = query.filter(ModelEvaluation.tenant_id == tenant_id)
                if status:
                    query = query.filter(ModelEvaluation.status == status)
                
                total = query.count()
                evaluations = query.order_by(
                    ModelEvaluation.created_at.desc()
                ).offset(offset).limit(limit).all()
                
                return [e.to_dict() for e in evaluations], total
                
        except Exception as e:
            logger.error(f"Failed to list evaluations by user: {e}")
            return [], 0
    
    def update(self, evaluation_id: str, update_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """更新评估记录"""
        try:
            if self._use_memory_storage:
                for record_id, eval_data in self._evaluations.items():
                    if eval_data.get('evaluation_id') == evaluation_id:
                        eval_data.update(update_data)
                        eval_data['updated_at'] = datetime.utcnow().isoformat()
                        return eval_data
                return None
            
            from backend.schemas.training_models import ModelEvaluation
            
            with self._db_manager.get_db_session() as db:
                evaluation = db.query(ModelEvaluation).filter(
                    ModelEvaluation.evaluation_id == evaluation_id
                ).first()
                
                if not evaluation:
                    return None
                
                for key, value in update_data.items():
                    if key in ('evaluation_config', 'metrics_summary', 'data_statistics', 
                              'tags', 'metadata_', 'error_details') and isinstance(value, (dict, list)):
                        value = json.dumps(value)
                    if hasattr(evaluation, key):
                        setattr(evaluation, key, value)
                
                evaluation.updated_at = datetime.utcnow()
                db.commit()
                db.refresh(evaluation)
                return evaluation.to_dict()
                
        except Exception as e:
            logger.error(f"Failed to update evaluation: {e}")
            return None
    
    def delete(self, evaluation_id: str) -> bool:
        """删除评估记录"""
        try:
            if self._use_memory_storage:
                for record_id, eval_data in list(self._evaluations.items()):
                    if eval_data.get('evaluation_id') == evaluation_id:
                        del self._evaluations[record_id]
                        # 删除关联的指标
                        metrics_to_delete = [
                            k for k, v in self._metrics.items()
                            if v.get('evaluation_id') == record_id
                        ]
                        for k in metrics_to_delete:
                            del self._metrics[k]
                        return True
                return False
            
            from backend.schemas.training_models import ModelEvaluation
            
            with self._db_manager.get_db_session() as db:
                evaluation = db.query(ModelEvaluation).filter(
                    ModelEvaluation.evaluation_id == evaluation_id
                ).first()
                
                if not evaluation:
                    return False
                
                db.delete(evaluation)
                db.commit()
                return True
                
        except Exception as e:
            logger.error(f"Failed to delete evaluation: {e}")
            return False
    
    def get_statistics(self, tenant_id: str, model_id: Optional[str] = None) -> Dict[str, Any]:
        """获取评估统计信息"""
        try:
            if self._use_memory_storage:
                evaluations = [
                    e for e in self._evaluations.values()
                    if e.get('tenant_id') == tenant_id
                ]
                if model_id:
                    evaluations = [e for e in evaluations if e.get('model_id') == model_id]
                
                status_counts = {}
                total_accuracy = 0
                accuracy_count = 0
                
                for e in evaluations:
                    status = e.get('status', 'unknown')
                    status_counts[status] = status_counts.get(status, 0) + 1
                    if e.get('accuracy') is not None:
                        total_accuracy += e['accuracy']
                        accuracy_count += 1
                
                return {
                    'total_evaluations': len(evaluations),
                    'by_status': status_counts,
                    'avg_accuracy': round(total_accuracy / accuracy_count, 4) if accuracy_count > 0 else 0,
                    'completed_count': status_counts.get('completed', 0),
                    'failed_count': status_counts.get('failed', 0)
                }
            
            from backend.schemas.training_models import ModelEvaluation
            from sqlalchemy import func as sql_func
            
            with self._db_manager.get_db_session() as db:
                query = db.query(ModelEvaluation).filter(
                    ModelEvaluation.tenant_id == tenant_id
                )
                if model_id:
                    query = query.filter(ModelEvaluation.model_id == model_id)
                
                total = query.count()
                
                # 按状态统计
                status_stats = db.query(
                    ModelEvaluation.status,
                    sql_func.count(ModelEvaluation.id)
                ).filter(
                    ModelEvaluation.tenant_id == tenant_id
                ).group_by(ModelEvaluation.status).all()
                
                # 平均准确率
                avg_accuracy = db.query(
                    sql_func.avg(ModelEvaluation.accuracy)
                ).filter(
                    ModelEvaluation.tenant_id == tenant_id,
                    ModelEvaluation.accuracy != None
                ).scalar() or 0
                
                return {
                    'total_evaluations': total,
                    'by_status': {r[0]: r[1] for r in status_stats},
                    'avg_accuracy': round(float(avg_accuracy), 4),
                    'completed_count': next((r[1] for r in status_stats if r[0] == 'completed'), 0),
                    'failed_count': next((r[1] for r in status_stats if r[0] == 'failed'), 0)
                }
                
        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {}
    
    # =========================================================================
    # 评估指标方法
    # =========================================================================
    
    def create_metric(self, metric_data: Dict[str, Any]) -> Dict[str, Any]:
        """创建评估指标"""
        try:
            metric_id = metric_data.get('id') or _generate_id()
            
            if self._use_memory_storage:
                metric_data['id'] = metric_id
                metric_data['created_at'] = datetime.utcnow().isoformat()
                self._metrics[metric_id] = metric_data
                return metric_data
            
            from backend.schemas.training_models import ModelEvaluationMetric
            
            with self._db_manager.get_db_session() as db:
                per_class_values = metric_data.get('per_class_values', {})
                if isinstance(per_class_values, dict):
                    per_class_values = json.dumps(per_class_values)
                
                additional_info = metric_data.get('additional_info', {})
                if isinstance(additional_info, dict):
                    additional_info = json.dumps(additional_info)
                
                metric = ModelEvaluationMetric(
                    id=metric_id,
                    evaluation_id=metric_data['evaluation_id'],
                    metric_name=metric_data['metric_name'],
                    metric_type=metric_data['metric_type'],
                    metric_value=metric_data['metric_value'],
                    confidence_lower=metric_data.get('confidence_lower'),
                    confidence_upper=metric_data.get('confidence_upper'),
                    confidence_level=metric_data.get('confidence_level', 0.95),
                    description=metric_data.get('description'),
                    unit=metric_data.get('unit'),
                    is_primary=metric_data.get('is_primary', False),
                    per_class_values=per_class_values,
                    additional_info=additional_info
                )
                db.add(metric)
                db.commit()
                db.refresh(metric)
                return metric.to_dict()
                
        except Exception as e:
            logger.error(f"Failed to create metric: {e}")
            raise
    
    def get_metrics_by_evaluation(self, evaluation_id: str) -> List[Dict[str, Any]]:
        """获取评估的所有指标"""
        try:
            if self._use_memory_storage:
                return [
                    m for m in self._metrics.values()
                    if str(m.get('evaluation_id')) == str(evaluation_id)
                ]
            
            from backend.schemas.training_models import ModelEvaluationMetric
            
            with self._db_manager.get_db_session() as db:
                metrics = db.query(ModelEvaluationMetric).filter(
                    ModelEvaluationMetric.evaluation_id == evaluation_id
                ).all()
                return [m.to_dict() for m in metrics]
                
        except Exception as e:
            logger.error(f"Failed to get metrics: {e}")
            return []


class ModelComparisonRepository:
    """模型对比数据访问层"""
    
    def __init__(self, use_memory_storage: bool = False):
        """初始化模型对比仓库"""
        self._use_memory_storage = use_memory_storage
        
        if use_memory_storage:
            self._comparisons: Dict[str, Dict] = {}
            self._db_manager = None
        else:
            try:
                from backend.modules.database.manager import get_database_manager
                self._db_manager = get_database_manager()
            except ImportError:
                logger.warning("Cannot import database service, falling back to memory storage")
                self._use_memory_storage = True
                self._comparisons: Dict[str, Dict] = {}
                self._db_manager = None
    
    def create(self, comparison_data: Dict[str, Any]) -> Dict[str, Any]:
        """创建对比记录"""
        try:
            record_id = comparison_data.get('id') or _generate_id()
            comparison_id = comparison_data.get('comparison_id') or f"cmp_{_generate_id()[:8]}"
            
            if self._use_memory_storage:
                comparison_data['id'] = record_id
                comparison_data['comparison_id'] = comparison_id
                comparison_data['created_at'] = datetime.utcnow().isoformat()
                comparison_data['updated_at'] = datetime.utcnow().isoformat()
                comparison_data.setdefault('status', 'pending')
                self._comparisons[record_id] = comparison_data
                return comparison_data
            
            from backend.schemas.training_models import ModelComparisonRecord
            
            with self._db_manager.get_db_session() as db:
                # 处理 JSON 字段
                model_ids = comparison_data.get('model_ids', [])
                if isinstance(model_ids, list):
                    model_ids = json.dumps(model_ids)
                
                comparison_config = comparison_data.get('comparison_config', {})
                if isinstance(comparison_config, dict):
                    comparison_config = json.dumps(comparison_config)
                
                comparison_metrics = comparison_data.get('comparison_metrics', [])
                if isinstance(comparison_metrics, list):
                    comparison_metrics = json.dumps(comparison_metrics)
                
                ranking = comparison_data.get('ranking', [])
                if isinstance(ranking, list):
                    ranking = json.dumps(ranking)
                
                recommendations = comparison_data.get('recommendations', [])
                if isinstance(recommendations, list):
                    recommendations = json.dumps(recommendations)
                
                risk_assessment = comparison_data.get('risk_assessment', {})
                if isinstance(risk_assessment, dict):
                    risk_assessment = json.dumps(risk_assessment)
                
                detailed_results = comparison_data.get('detailed_results', {})
                if isinstance(detailed_results, dict):
                    detailed_results = json.dumps(detailed_results)
                
                tags = comparison_data.get('tags', [])
                if isinstance(tags, list):
                    tags = json.dumps(tags)
                
                metadata = comparison_data.get('metadata', {}) or comparison_data.get('metadata_', {})
                if isinstance(metadata, dict):
                    metadata = json.dumps(metadata)
                
                comparison = ModelComparisonRecord(
                    id=record_id,
                    tenant_id=comparison_data.get('tenant_id'),
                    comparison_id=comparison_id,
                    user_id=comparison_data.get('user_id'),
                    dataset_id=comparison_data['dataset_id'],
                    model_ids=model_ids,
                    winner_model_id=comparison_data.get('winner_model_id'),
                    status=comparison_data.get('status', 'pending'),
                    comparison_config=comparison_config,
                    comparison_metrics=comparison_metrics,
                    decision_criteria=comparison_data.get('decision_criteria', 'multi_objective'),
                    ranking=ranking,
                    recommendations=recommendations,
                    risk_assessment=risk_assessment,
                    detailed_results=detailed_results,
                    started_at=comparison_data.get('started_at'),
                    completed_at=comparison_data.get('completed_at'),
                    duration_seconds=comparison_data.get('duration_seconds'),
                    error_message=comparison_data.get('error_message'),
                    tags=tags,
                    metadata_=metadata
                )
                db.add(comparison)
                db.commit()
                db.refresh(comparison)
                return comparison.to_dict()
                
        except Exception as e:
            logger.error(f"Failed to create comparison: {e}")
            raise
    
    def get_by_id(self, comparison_id: str) -> Optional[Dict[str, Any]]:
        """根据对比ID获取记录"""
        try:
            if self._use_memory_storage:
                for cmp_data in self._comparisons.values():
                    if cmp_data.get('comparison_id') == comparison_id:
                        return cmp_data
                return None
            
            from backend.schemas.training_models import ModelComparisonRecord
            
            with self._db_manager.get_db_session() as db:
                comparison = db.query(ModelComparisonRecord).filter(
                    ModelComparisonRecord.comparison_id == comparison_id
                ).first()
                return comparison.to_dict() if comparison else None
                
        except Exception as e:
            logger.error(f"Failed to get comparison: {e}")
            return None
    
    def list_by_user(
        self,
        user_id: str,
        tenant_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> Tuple[List[Dict[str, Any]], int]:
        """获取用户的对比记录列表"""
        try:
            if self._use_memory_storage:
                comparisons = [
                    c for c in self._comparisons.values()
                    if c.get('user_id') == user_id
                ]
                if tenant_id:
                    comparisons = [c for c in comparisons if c.get('tenant_id') == tenant_id]
                if status:
                    comparisons = [c for c in comparisons if c.get('status') == status]
                
                comparisons.sort(key=lambda x: x.get('created_at', ''), reverse=True)
                total = len(comparisons)
                return comparisons[offset:offset + limit], total
            
            from backend.schemas.training_models import ModelComparisonRecord
            
            with self._db_manager.get_db_session() as db:
                query = db.query(ModelComparisonRecord).filter(
                    ModelComparisonRecord.user_id == user_id
                )
                if tenant_id:
                    query = query.filter(ModelComparisonRecord.tenant_id == tenant_id)
                if status:
                    query = query.filter(ModelComparisonRecord.status == status)
                
                total = query.count()
                comparisons = query.order_by(
                    ModelComparisonRecord.created_at.desc()
                ).offset(offset).limit(limit).all()
                
                return [c.to_dict() for c in comparisons], total
                
        except Exception as e:
            logger.error(f"Failed to list comparisons: {e}")
            return [], 0
    
    def update(self, comparison_id: str, update_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """更新对比记录"""
        try:
            if self._use_memory_storage:
                for record_id, cmp_data in self._comparisons.items():
                    if cmp_data.get('comparison_id') == comparison_id:
                        cmp_data.update(update_data)
                        cmp_data['updated_at'] = datetime.utcnow().isoformat()
                        return cmp_data
                return None
            
            from backend.schemas.training_models import ModelComparisonRecord
            
            with self._db_manager.get_db_session() as db:
                comparison = db.query(ModelComparisonRecord).filter(
                    ModelComparisonRecord.comparison_id == comparison_id
                ).first()
                
                if not comparison:
                    return None
                
                for key, value in update_data.items():
                    if key in ('model_ids', 'comparison_config', 'comparison_metrics', 
                              'ranking', 'recommendations', 'risk_assessment', 
                              'detailed_results', 'tags', 'metadata_') and isinstance(value, (dict, list)):
                        value = json.dumps(value)
                    if hasattr(comparison, key):
                        setattr(comparison, key, value)
                
                comparison.updated_at = datetime.utcnow()
                db.commit()
                db.refresh(comparison)
                return comparison.to_dict()
                
        except Exception as e:
            logger.error(f"Failed to update comparison: {e}")
            return None
    
    def delete(self, comparison_id: str) -> bool:
        """删除对比记录"""
        try:
            if self._use_memory_storage:
                for record_id, cmp_data in list(self._comparisons.items()):
                    if cmp_data.get('comparison_id') == comparison_id:
                        del self._comparisons[record_id]
                        return True
                return False
            
            from backend.schemas.training_models import ModelComparisonRecord
            
            with self._db_manager.get_db_session() as db:
                comparison = db.query(ModelComparisonRecord).filter(
                    ModelComparisonRecord.comparison_id == comparison_id
                ).first()
                
                if not comparison:
                    return False
                
                db.delete(comparison)
                db.commit()
                return True
                
        except Exception as e:
            logger.error(f"Failed to delete comparison: {e}")
            return False


# 全局仓库实例
_evaluation_repository = None
_comparison_repository = None


def get_model_evaluation_repository(use_memory_storage: bool = False) -> ModelEvaluationRepository:
    """获取模型评估仓库实例"""
    global _evaluation_repository
    if _evaluation_repository is None:
        _evaluation_repository = ModelEvaluationRepository(use_memory_storage=use_memory_storage)
    return _evaluation_repository


def get_model_comparison_repository(use_memory_storage: bool = False) -> ModelComparisonRepository:
    """获取模型对比仓库实例"""
    global _comparison_repository
    if _comparison_repository is None:
        _comparison_repository = ModelComparisonRepository(use_memory_storage=use_memory_storage)
    return _comparison_repository


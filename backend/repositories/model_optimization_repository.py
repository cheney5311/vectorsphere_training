"""模型优化数据访问层

提供模型优化相关的数据库访问功能，包括：
- 模型压缩记录
- 推理优化记录
- 自动优化记录
"""

import json
import logging
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime
import uuid

logger = logging.getLogger(__name__)


def _generate_id() -> str:
    """生成唯一ID"""
    return str(uuid.uuid4())


class ModelOptimizationRepository:
    """模型优化数据访问层"""
    
    def __init__(self, use_memory_storage: bool = False):
        """初始化模型优化仓库
        
        Args:
            use_memory_storage: 是否使用内存存储
        """
        self._use_memory_storage = use_memory_storage
        
        if use_memory_storage:
            self._optimizations: Dict[str, Dict] = {}
            self._db_manager = None
        else:
            try:
                from backend.modules.database.manager import get_database_manager
                self._db_manager = get_database_manager()
            except ImportError:
                logger.warning("Cannot import database service, falling back to memory storage")
                self._use_memory_storage = True
                self._optimizations: Dict[str, Dict] = {}
                self._db_manager = None
    
    def create(self, optimization_data: Dict[str, Any]) -> Dict[str, Any]:
        """创建优化记录
        
        Args:
            optimization_data: 优化数据
            
        Returns:
            创建的优化记录
        """
        try:
            record_id = optimization_data.get('id') or _generate_id()
            optimization_id = optimization_data.get('optimization_id') or f"opt_{_generate_id()[:8]}"
            
            if self._use_memory_storage:
                optimization_data['id'] = record_id
                optimization_data['optimization_id'] = optimization_id
                optimization_data['created_at'] = datetime.utcnow().isoformat()
                optimization_data['updated_at'] = datetime.utcnow().isoformat()
                optimization_data.setdefault('status', 'pending')
                self._optimizations[record_id] = optimization_data
                return optimization_data
            
            from backend.schemas.training_models import ModelOptimization
            
            with self._db_manager.get_db_session() as db:
                # 处理 JSON 字段
                optimization_config = optimization_data.get('optimization_config', {})
                if isinstance(optimization_config, dict):
                    optimization_config = json.dumps(optimization_config)
                
                target_constraints = optimization_data.get('target_constraints', {})
                if isinstance(target_constraints, dict):
                    target_constraints = json.dumps(target_constraints)
                
                metrics = optimization_data.get('metrics', {})
                if isinstance(metrics, dict):
                    metrics = json.dumps(metrics)
                
                error_details = optimization_data.get('error_details', {})
                if isinstance(error_details, dict):
                    error_details = json.dumps(error_details)
                
                tags = optimization_data.get('tags', [])
                if isinstance(tags, list):
                    tags = json.dumps(tags)
                
                metadata = optimization_data.get('metadata', {}) or optimization_data.get('metadata_', {})
                if isinstance(metadata, dict):
                    metadata = json.dumps(metadata)
                
                optimization = ModelOptimization(
                    id=record_id,
                    tenant_id=optimization_data.get('tenant_id'),
                    optimization_id=optimization_id,
                    original_model_id=optimization_data['original_model_id'],
                    optimized_model_id=optimization_data.get('optimized_model_id'),
                    user_id=optimization_data.get('user_id'),
                    optimization_type=optimization_data.get('optimization_type', 'compression'),
                    technique=optimization_data.get('technique'),
                    strategy=optimization_data.get('strategy'),
                    status=optimization_data.get('status', 'pending'),
                    compression_ratio=optimization_data.get('compression_ratio'),
                    quantization_bits=optimization_data.get('quantization_bits'),
                    preserve_accuracy=optimization_data.get('preserve_accuracy', True),
                    hardware_target=optimization_data.get('hardware_target'),
                    graph_optimization=optimization_data.get('graph_optimization', False),
                    operator_fusion=optimization_data.get('operator_fusion', False),
                    constant_folding=optimization_data.get('constant_folding', False),
                    dead_code_elimination=optimization_data.get('dead_code_elimination', False),
                    memory_optimization=optimization_data.get('memory_optimization', False),
                    accuracy_preserved=optimization_data.get('accuracy_preserved'),
                    model_size_reduction=optimization_data.get('model_size_reduction'),
                    inference_speedup=optimization_data.get('inference_speedup'),
                    original_size_mb=optimization_data.get('original_size_mb'),
                    optimized_size_mb=optimization_data.get('optimized_size_mb'),
                    latency_reduction=optimization_data.get('latency_reduction'),
                    memory_usage_reduction=optimization_data.get('memory_usage_reduction'),
                    throughput_improvement=optimization_data.get('throughput_improvement'),
                    original_latency_ms=optimization_data.get('original_latency_ms'),
                    optimized_latency_ms=optimization_data.get('optimized_latency_ms'),
                    started_at=optimization_data.get('started_at'),
                    completed_at=optimization_data.get('completed_at'),
                    optimization_time_seconds=optimization_data.get('optimization_time_seconds'),
                    optimization_config=optimization_config,
                    target_constraints=target_constraints,
                    metrics=metrics,
                    error_message=optimization_data.get('error_message'),
                    error_details=error_details,
                    tags=tags,
                    metadata_=metadata
                )
                db.add(optimization)
                db.commit()
                db.refresh(optimization)
                return optimization.to_dict()
                
        except Exception as e:
            logger.error(f"Failed to create optimization: {e}")
            raise
    
    def get_by_id(self, optimization_id: str) -> Optional[Dict[str, Any]]:
        """根据优化ID获取记录"""
        try:
            if self._use_memory_storage:
                for opt_data in self._optimizations.values():
                    if opt_data.get('optimization_id') == optimization_id:
                        return opt_data
                return None
            
            from backend.schemas.training_models import ModelOptimization
            
            with self._db_manager.get_db_session() as db:
                optimization = db.query(ModelOptimization).filter(
                    ModelOptimization.optimization_id == optimization_id
                ).first()
                return optimization.to_dict() if optimization else None
                
        except Exception as e:
            logger.error(f"Failed to get optimization: {e}")
            return None
    
    def list_by_model(
        self,
        original_model_id: str,
        tenant_id: Optional[str] = None,
        optimization_type: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> Tuple[List[Dict[str, Any]], int]:
        """获取模型的优化记录列表"""
        try:
            if self._use_memory_storage:
                optimizations = [
                    o for o in self._optimizations.values()
                    if o.get('original_model_id') == original_model_id
                ]
                if tenant_id:
                    optimizations = [o for o in optimizations if o.get('tenant_id') == tenant_id]
                if optimization_type:
                    optimizations = [o for o in optimizations if o.get('optimization_type') == optimization_type]
                if status:
                    optimizations = [o for o in optimizations if o.get('status') == status]
                
                optimizations.sort(key=lambda x: x.get('created_at', ''), reverse=True)
                total = len(optimizations)
                return optimizations[offset:offset + limit], total
            
            from backend.schemas.training_models import ModelOptimization
            
            with self._db_manager.get_db_session() as db:
                query = db.query(ModelOptimization).filter(
                    ModelOptimization.original_model_id == original_model_id
                )
                if tenant_id:
                    query = query.filter(ModelOptimization.tenant_id == tenant_id)
                if optimization_type:
                    query = query.filter(ModelOptimization.optimization_type == optimization_type)
                if status:
                    query = query.filter(ModelOptimization.status == status)
                
                total = query.count()
                optimizations = query.order_by(
                    ModelOptimization.created_at.desc()
                ).offset(offset).limit(limit).all()
                
                return [o.to_dict() for o in optimizations], total
                
        except Exception as e:
            logger.error(f"Failed to list optimizations by model: {e}")
            return [], 0
    
    def list_by_user(
        self,
        user_id: str,
        tenant_id: Optional[str] = None,
        optimization_type: Optional[str] = None,
        technique: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> Tuple[List[Dict[str, Any]], int]:
        """获取用户的优化记录列表"""
        try:
            if self._use_memory_storage:
                optimizations = [
                    o for o in self._optimizations.values()
                    if o.get('user_id') == user_id
                ]
                if tenant_id:
                    optimizations = [o for o in optimizations if o.get('tenant_id') == tenant_id]
                if optimization_type:
                    optimizations = [o for o in optimizations if o.get('optimization_type') == optimization_type]
                if technique:
                    optimizations = [o for o in optimizations if o.get('technique') == technique]
                if status:
                    optimizations = [o for o in optimizations if o.get('status') == status]
                
                optimizations.sort(key=lambda x: x.get('created_at', ''), reverse=True)
                total = len(optimizations)
                return optimizations[offset:offset + limit], total
            
            from backend.schemas.training_models import ModelOptimization
            
            with self._db_manager.get_db_session() as db:
                query = db.query(ModelOptimization).filter(
                    ModelOptimization.user_id == user_id
                )
                if tenant_id:
                    query = query.filter(ModelOptimization.tenant_id == tenant_id)
                if optimization_type:
                    query = query.filter(ModelOptimization.optimization_type == optimization_type)
                if technique:
                    query = query.filter(ModelOptimization.technique == technique)
                if status:
                    query = query.filter(ModelOptimization.status == status)
                
                total = query.count()
                optimizations = query.order_by(
                    ModelOptimization.created_at.desc()
                ).offset(offset).limit(limit).all()
                
                return [o.to_dict() for o in optimizations], total
                
        except Exception as e:
            logger.error(f"Failed to list optimizations by user: {e}")
            return [], 0
    
    def update(self, optimization_id: str, update_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """更新优化记录"""
        try:
            if self._use_memory_storage:
                for record_id, opt_data in self._optimizations.items():
                    if opt_data.get('optimization_id') == optimization_id:
                        opt_data.update(update_data)
                        opt_data['updated_at'] = datetime.utcnow().isoformat()
                        return opt_data
                return None
            
            from backend.schemas.training_models import ModelOptimization
            
            with self._db_manager.get_db_session() as db:
                optimization = db.query(ModelOptimization).filter(
                    ModelOptimization.optimization_id == optimization_id
                ).first()
                
                if not optimization:
                    return None
                
                for key, value in update_data.items():
                    if key in ('optimization_config', 'target_constraints', 'metrics',
                              'error_details', 'tags', 'metadata_') and isinstance(value, (dict, list)):
                        value = json.dumps(value)
                    if hasattr(optimization, key):
                        setattr(optimization, key, value)
                
                optimization.updated_at = datetime.utcnow()
                db.commit()
                db.refresh(optimization)
                return optimization.to_dict()
                
        except Exception as e:
            logger.error(f"Failed to update optimization: {e}")
            return None
    
    def delete(self, optimization_id: str) -> bool:
        """删除优化记录"""
        try:
            if self._use_memory_storage:
                for record_id, opt_data in list(self._optimizations.items()):
                    if opt_data.get('optimization_id') == optimization_id:
                        del self._optimizations[record_id]
                        return True
                return False
            
            from backend.schemas.training_models import ModelOptimization
            
            with self._db_manager.get_db_session() as db:
                optimization = db.query(ModelOptimization).filter(
                    ModelOptimization.optimization_id == optimization_id
                ).first()
                
                if not optimization:
                    return False
                
                db.delete(optimization)
                db.commit()
                return True
                
        except Exception as e:
            logger.error(f"Failed to delete optimization: {e}")
            return False
    
    def get_statistics(
        self,
        tenant_id: str,
        user_id: Optional[str] = None,
        model_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """获取优化统计信息"""
        try:
            if self._use_memory_storage:
                optimizations = [
                    o for o in self._optimizations.values()
                    if o.get('tenant_id') == tenant_id
                ]
                if user_id:
                    optimizations = [o for o in optimizations if o.get('user_id') == user_id]
                if model_id:
                    optimizations = [o for o in optimizations if o.get('original_model_id') == model_id]
                
                status_counts = {}
                type_counts = {}
                technique_counts = {}
                total_size_reduction = 0
                total_speedup = 0
                count_with_metrics = 0
                
                for o in optimizations:
                    status = o.get('status', 'unknown')
                    status_counts[status] = status_counts.get(status, 0) + 1
                    
                    opt_type = o.get('optimization_type', 'unknown')
                    type_counts[opt_type] = type_counts.get(opt_type, 0) + 1
                    
                    technique = o.get('technique', 'unknown')
                    if technique:
                        technique_counts[technique] = technique_counts.get(technique, 0) + 1
                    
                    if o.get('model_size_reduction') is not None:
                        total_size_reduction += o['model_size_reduction']
                        count_with_metrics += 1
                    
                    if o.get('inference_speedup') is not None:
                        total_speedup += o['inference_speedup']
                
                return {
                    'total_optimizations': len(optimizations),
                    'by_status': status_counts,
                    'by_type': type_counts,
                    'by_technique': technique_counts,
                    'completed_count': status_counts.get('completed', 0),
                    'failed_count': status_counts.get('failed', 0),
                    'avg_size_reduction': round(total_size_reduction / count_with_metrics, 4) if count_with_metrics > 0 else 0,
                    'avg_speedup': round(total_speedup / count_with_metrics, 4) if count_with_metrics > 0 else 0
                }
            
            from backend.schemas.training_models import ModelOptimization
            from sqlalchemy import func as sql_func
            
            with self._db_manager.get_db_session() as db:
                query = db.query(ModelOptimization).filter(
                    ModelOptimization.tenant_id == tenant_id
                )
                if user_id:
                    query = query.filter(ModelOptimization.user_id == user_id)
                if model_id:
                    query = query.filter(ModelOptimization.original_model_id == model_id)
                
                total = query.count()
                
                # 按状态统计
                status_stats = db.query(
                    ModelOptimization.status,
                    sql_func.count(ModelOptimization.id)
                ).filter(
                    ModelOptimization.tenant_id == tenant_id
                ).group_by(ModelOptimization.status).all()
                
                # 按类型统计
                type_stats = db.query(
                    ModelOptimization.optimization_type,
                    sql_func.count(ModelOptimization.id)
                ).filter(
                    ModelOptimization.tenant_id == tenant_id
                ).group_by(ModelOptimization.optimization_type).all()
                
                # 平均指标
                avg_size_reduction = db.query(
                    sql_func.avg(ModelOptimization.model_size_reduction)
                ).filter(
                    ModelOptimization.tenant_id == tenant_id,
                    ModelOptimization.model_size_reduction != None
                ).scalar() or 0
                
                avg_speedup = db.query(
                    sql_func.avg(ModelOptimization.inference_speedup)
                ).filter(
                    ModelOptimization.tenant_id == tenant_id,
                    ModelOptimization.inference_speedup != None
                ).scalar() or 0
                
                return {
                    'total_optimizations': total,
                    'by_status': {r[0]: r[1] for r in status_stats},
                    'by_type': {r[0]: r[1] for r in type_stats},
                    'completed_count': next((r[1] for r in status_stats if r[0] == 'completed'), 0),
                    'failed_count': next((r[1] for r in status_stats if r[0] == 'failed'), 0),
                    'avg_size_reduction': round(float(avg_size_reduction), 4),
                    'avg_speedup': round(float(avg_speedup), 4)
                }
                
        except Exception as e:
            logger.error(f"Failed to get optimization statistics: {e}")
            return {}
    
    def get_best_optimization(
        self,
        model_id: str,
        tenant_id: Optional[str] = None,
        optimization_type: Optional[str] = None,
        metric: str = 'inference_speedup'
    ) -> Optional[Dict[str, Any]]:
        """获取模型的最佳优化记录"""
        try:
            if self._use_memory_storage:
                optimizations = [
                    o for o in self._optimizations.values()
                    if o.get('original_model_id') == model_id
                    and o.get('status') == 'completed'
                ]
                if tenant_id:
                    optimizations = [o for o in optimizations if o.get('tenant_id') == tenant_id]
                if optimization_type:
                    optimizations = [o for o in optimizations if o.get('optimization_type') == optimization_type]
                
                if not optimizations:
                    return None
                
                # 按指定指标排序
                optimizations.sort(
                    key=lambda x: x.get(metric, 0) or 0,
                    reverse=True
                )
                return optimizations[0]
            
            from backend.schemas.training_models import ModelOptimization
            
            with self._db_manager.get_db_session() as db:
                query = db.query(ModelOptimization).filter(
                    ModelOptimization.original_model_id == model_id,
                    ModelOptimization.status == 'completed'
                )
                if tenant_id:
                    query = query.filter(ModelOptimization.tenant_id == tenant_id)
                if optimization_type:
                    query = query.filter(ModelOptimization.optimization_type == optimization_type)
                
                # 按指定指标排序
                metric_column = getattr(ModelOptimization, metric, ModelOptimization.inference_speedup)
                optimization = query.order_by(metric_column.desc()).first()
                
                return optimization.to_dict() if optimization else None
                
        except Exception as e:
            logger.error(f"Failed to get best optimization: {e}")
            return None


# 全局仓库实例
_optimization_repository = None


def get_model_optimization_repository(use_memory_storage: bool = False) -> ModelOptimizationRepository:
    """获取模型优化仓库实例"""
    global _optimization_repository
    if _optimization_repository is None:
        _optimization_repository = ModelOptimizationRepository(use_memory_storage=use_memory_storage)
    return _optimization_repository




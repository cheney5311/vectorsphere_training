"""模型选择数据访问层

提供模型选择相关的数据库访问功能，包括：
- 模型推荐记录
- 模型配置记录
- 模型目录管理
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


class ModelRecommendationRepository:
    """模型推荐记录数据访问层"""
    
    def __init__(self, use_memory_storage: bool = False):
        """初始化模型推荐仓库
        
        Args:
            use_memory_storage: 是否使用内存存储
        """
        self._use_memory_storage = use_memory_storage
        
        if use_memory_storage:
            self._recommendations: Dict[str, Dict] = {}
            self._db_manager = None
        else:
            try:
                from backend.modules.database.manager import get_database_manager
                self._db_manager = get_database_manager()
            except ImportError:
                logger.warning("Cannot import database service, falling back to memory storage")
                self._use_memory_storage = True
                self._recommendations: Dict[str, Dict] = {}
                self._db_manager = None
    
    def create(self, recommendation_data: Dict[str, Any]) -> Dict[str, Any]:
        """创建推荐记录
        
        Args:
            recommendation_data: 推荐数据
            
        Returns:
            创建的推荐记录
        """
        try:
            record_id = recommendation_data.get('id') or _generate_id()
            recommendation_id = recommendation_data.get('recommendation_id') or f"rec_{_generate_id()[:8]}"
            
            if self._use_memory_storage:
                recommendation_data['id'] = record_id
                recommendation_data['recommendation_id'] = recommendation_id
                recommendation_data['created_at'] = datetime.utcnow().isoformat()
                recommendation_data['updated_at'] = datetime.utcnow().isoformat()
                recommendation_data.setdefault('status', 'completed')
                self._recommendations[record_id] = recommendation_data
                return recommendation_data
            
            from backend.schemas.training_models import ModelRecommendationRecord
            
            with self._db_manager.get_db_session() as db:
                recommendation = ModelRecommendationRecord(
                    id=record_id,
                    recommendation_id=recommendation_id,
                    tenant_id=recommendation_data.get('tenant_id'),
                    user_id=recommendation_data.get('user_id'),
                    task_type=recommendation_data['task_type'],
                    requirements=json.dumps(recommendation_data.get('requirements', {})) if recommendation_data.get('requirements') else None,
                    performance_requirements=json.dumps(recommendation_data.get('performance_requirements', {})) if recommendation_data.get('performance_requirements') else None,
                    recommended_models=json.dumps(recommendation_data.get('recommended_models', [])),
                    top_recommendation=recommendation_data.get('top_recommendation'),
                    top_confidence=recommendation_data.get('top_confidence'),
                    num_recommendations=recommendation_data.get('num_recommendations', 0),
                    selected_model=recommendation_data.get('selected_model'),
                    feedback_score=recommendation_data.get('feedback_score'),
                    feedback_comment=recommendation_data.get('feedback_comment'),
                    is_helpful=recommendation_data.get('is_helpful'),
                    status=recommendation_data.get('status', 'completed'),
                    response_time_ms=recommendation_data.get('response_time_ms'),
                    source=recommendation_data.get('source', 'api'),
                    metadata_=json.dumps(recommendation_data.get('metadata', {})) if recommendation_data.get('metadata') else None
                )
                db.add(recommendation)
                db.commit()
                db.refresh(recommendation)
                return recommendation.to_dict()
                
        except Exception as e:
            logger.error(f"Failed to create recommendation record: {e}")
            raise
    
    def get_by_id(self, recommendation_id: str) -> Optional[Dict[str, Any]]:
        """根据推荐ID获取记录"""
        try:
            if self._use_memory_storage:
                for rec_data in self._recommendations.values():
                    if rec_data.get('recommendation_id') == recommendation_id:
                        return rec_data
                return None
            
            from backend.schemas.training_models import ModelRecommendationRecord
            
            with self._db_manager.get_db_session() as db:
                recommendation = db.query(ModelRecommendationRecord).filter(
                    ModelRecommendationRecord.recommendation_id == recommendation_id
                ).first()
                return recommendation.to_dict() if recommendation else None
                
        except Exception as e:
            logger.error(f"Failed to get recommendation: {e}")
            return None
    
    def list_by_user(
        self,
        user_id: str,
        tenant_id: Optional[str] = None,
        task_type: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> Tuple[List[Dict[str, Any]], int]:
        """获取用户的推荐记录列表"""
        try:
            if self._use_memory_storage:
                recommendations = [
                    r for r in self._recommendations.values()
                    if r.get('user_id') == user_id
                ]
                if tenant_id:
                    recommendations = [r for r in recommendations if r.get('tenant_id') == tenant_id]
                if task_type:
                    recommendations = [r for r in recommendations if r.get('task_type') == task_type]
                
                recommendations.sort(key=lambda x: x.get('created_at', ''), reverse=True)
                total = len(recommendations)
                return recommendations[offset:offset + limit], total
            
            from backend.schemas.training_models import ModelRecommendationRecord
            
            with self._db_manager.get_db_session() as db:
                query = db.query(ModelRecommendationRecord).filter(
                    ModelRecommendationRecord.user_id == user_id
                )
                if tenant_id:
                    query = query.filter(ModelRecommendationRecord.tenant_id == tenant_id)
                if task_type:
                    query = query.filter(ModelRecommendationRecord.task_type == task_type)
                
                total = query.count()
                recommendations = query.order_by(
                    ModelRecommendationRecord.created_at.desc()
                ).offset(offset).limit(limit).all()
                
                return [r.to_dict() for r in recommendations], total
                
        except Exception as e:
            logger.error(f"Failed to list recommendations: {e}")
            return [], 0
    
    def update(self, recommendation_id: str, update_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """更新推荐记录（用于用户反馈）"""
        try:
            if self._use_memory_storage:
                for record_id, rec_data in self._recommendations.items():
                    if rec_data.get('recommendation_id') == recommendation_id:
                        rec_data.update(update_data)
                        rec_data['updated_at'] = datetime.utcnow().isoformat()
                        return rec_data
                return None
            
            from backend.schemas.training_models import ModelRecommendationRecord
            
            with self._db_manager.get_db_session() as db:
                recommendation = db.query(ModelRecommendationRecord).filter(
                    ModelRecommendationRecord.recommendation_id == recommendation_id
                ).first()
                
                if not recommendation:
                    return None
                
                for key, value in update_data.items():
                    if key in ('requirements', 'performance_requirements', 'recommended_models', 'metadata_') and isinstance(value, (dict, list)):
                        value = json.dumps(value)
                    if hasattr(recommendation, key):
                        setattr(recommendation, key, value)
                
                recommendation.updated_at = datetime.utcnow()
                db.commit()
                db.refresh(recommendation)
                return recommendation.to_dict()
                
        except Exception as e:
            logger.error(f"Failed to update recommendation: {e}")
            return None
    
    def get_statistics(
        self,
        tenant_id: str,
        user_id: Optional[str] = None,
        task_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """获取推荐统计信息"""
        try:
            if self._use_memory_storage:
                recommendations = [
                    r for r in self._recommendations.values()
                    if r.get('tenant_id') == tenant_id
                ]
                if user_id:
                    recommendations = [r for r in recommendations if r.get('user_id') == user_id]
                if task_type:
                    recommendations = [r for r in recommendations if r.get('task_type') == task_type]
                
                task_counts = {}
                total_confidence = 0
                helpful_count = 0
                feedback_count = 0
                
                for r in recommendations:
                    task = r.get('task_type', 'unknown')
                    task_counts[task] = task_counts.get(task, 0) + 1
                    
                    if r.get('top_confidence'):
                        total_confidence += r['top_confidence']
                    
                    if r.get('is_helpful') is not None:
                        feedback_count += 1
                        if r['is_helpful']:
                            helpful_count += 1
                
                return {
                    'total_recommendations': len(recommendations),
                    'by_task_type': task_counts,
                    'avg_confidence': round(total_confidence / len(recommendations), 4) if recommendations else 0,
                    'helpful_rate': round(helpful_count / feedback_count, 4) if feedback_count > 0 else None,
                    'feedback_count': feedback_count
                }
            
            from backend.schemas.training_models import ModelRecommendationRecord
            from sqlalchemy import func as sql_func
            
            with self._db_manager.get_db_session() as db:
                query = db.query(ModelRecommendationRecord).filter(
                    ModelRecommendationRecord.tenant_id == tenant_id
                )
                if user_id:
                    query = query.filter(ModelRecommendationRecord.user_id == user_id)
                if task_type:
                    query = query.filter(ModelRecommendationRecord.task_type == task_type)
                
                total = query.count()
                
                avg_confidence = db.query(
                    sql_func.avg(ModelRecommendationRecord.top_confidence)
                ).filter(
                    ModelRecommendationRecord.tenant_id == tenant_id,
                    ModelRecommendationRecord.top_confidence != None
                ).scalar() or 0
                
                return {
                    'total_recommendations': total,
                    'avg_confidence': round(float(avg_confidence), 4)
                }
                
        except Exception as e:
            logger.error(f"Failed to get recommendation statistics: {e}")
            return {}


class ModelConfigurationRepository:
    """模型配置记录数据访问层"""
    
    def __init__(self, use_memory_storage: bool = False):
        """初始化模型配置仓库"""
        self._use_memory_storage = use_memory_storage
        
        if use_memory_storage:
            self._configurations: Dict[str, Dict] = {}
            self._db_manager = None
        else:
            try:
                from backend.modules.database.manager import get_database_manager
                self._db_manager = get_database_manager()
            except ImportError:
                logger.warning("Cannot import database service, falling back to memory storage")
                self._use_memory_storage = True
                self._configurations: Dict[str, Dict] = {}
                self._db_manager = None
    
    def create(self, configuration_data: Dict[str, Any]) -> Dict[str, Any]:
        """创建配置记录"""
        try:
            record_id = configuration_data.get('id') or _generate_id()
            configuration_id = configuration_data.get('configuration_id') or f"cfg_{_generate_id()[:8]}"
            
            if self._use_memory_storage:
                configuration_data['id'] = record_id
                configuration_data['configuration_id'] = configuration_id
                configuration_data['created_at'] = datetime.utcnow().isoformat()
                configuration_data['updated_at'] = datetime.utcnow().isoformat()
                configuration_data.setdefault('status', 'active')
                self._configurations[record_id] = configuration_data
                return configuration_data
            
            from backend.schemas.training_models import ModelConfigurationRecord
            
            with self._db_manager.get_db_session() as db:
                configuration = ModelConfigurationRecord(
                    id=record_id,
                    configuration_id=configuration_id,
                    tenant_id=configuration_data.get('tenant_id'),
                    user_id=configuration_data.get('user_id'),
                    model_name=configuration_data['model_name'],
                    task_type=configuration_data['task_type'],
                    framework=configuration_data.get('framework'),
                    model_type=configuration_data.get('model_type'),
                    dataset_id=configuration_data.get('dataset_id'),
                    dataset_info=json.dumps(configuration_data.get('dataset_info', {})) if configuration_data.get('dataset_info') else None,
                    hyperparameters=json.dumps(configuration_data.get('hyperparameters', {})) if configuration_data.get('hyperparameters') else None,
                    training_config=json.dumps(configuration_data.get('training_config', {})) if configuration_data.get('training_config') else None,
                    hardware_config=json.dumps(configuration_data.get('hardware_config', {})) if configuration_data.get('hardware_config') else None,
                    full_config=json.dumps(configuration_data.get('full_config', {})) if configuration_data.get('full_config') else None,
                    config_source=configuration_data.get('config_source', 'auto'),
                    template_id=configuration_data.get('template_id'),
                    status=configuration_data.get('status', 'active'),
                    is_default=configuration_data.get('is_default', False),
                    usage_count=configuration_data.get('usage_count', 0),
                    tags=json.dumps(configuration_data.get('tags', [])) if configuration_data.get('tags') else None,
                    metadata_=json.dumps(configuration_data.get('metadata', {})) if configuration_data.get('metadata') else None
                )
                db.add(configuration)
                db.commit()
                db.refresh(configuration)
                return configuration.to_dict()
                
        except Exception as e:
            logger.error(f"Failed to create configuration record: {e}")
            raise
    
    def get_by_id(self, configuration_id: str) -> Optional[Dict[str, Any]]:
        """根据配置ID获取记录"""
        try:
            if self._use_memory_storage:
                for cfg_data in self._configurations.values():
                    if cfg_data.get('configuration_id') == configuration_id:
                        return cfg_data
                return None
            
            from backend.schemas.training_models import ModelConfigurationRecord
            
            with self._db_manager.get_db_session() as db:
                configuration = db.query(ModelConfigurationRecord).filter(
                    ModelConfigurationRecord.configuration_id == configuration_id
                ).first()
                return configuration.to_dict() if configuration else None
                
        except Exception as e:
            logger.error(f"Failed to get configuration: {e}")
            return None
    
    def list_by_user(
        self,
        user_id: str,
        tenant_id: Optional[str] = None,
        model_name: Optional[str] = None,
        task_type: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> Tuple[List[Dict[str, Any]], int]:
        """获取用户的配置记录列表"""
        try:
            if self._use_memory_storage:
                configurations = [
                    c for c in self._configurations.values()
                    if c.get('user_id') == user_id
                ]
                if tenant_id:
                    configurations = [c for c in configurations if c.get('tenant_id') == tenant_id]
                if model_name:
                    configurations = [c for c in configurations if c.get('model_name') == model_name]
                if task_type:
                    configurations = [c for c in configurations if c.get('task_type') == task_type]
                
                configurations.sort(key=lambda x: x.get('created_at', ''), reverse=True)
                total = len(configurations)
                return configurations[offset:offset + limit], total
            
            from backend.schemas.training_models import ModelConfigurationRecord
            
            with self._db_manager.get_db_session() as db:
                query = db.query(ModelConfigurationRecord).filter(
                    ModelConfigurationRecord.user_id == user_id
                )
                if tenant_id:
                    query = query.filter(ModelConfigurationRecord.tenant_id == tenant_id)
                if model_name:
                    query = query.filter(ModelConfigurationRecord.model_name == model_name)
                if task_type:
                    query = query.filter(ModelConfigurationRecord.task_type == task_type)
                
                total = query.count()
                configurations = query.order_by(
                    ModelConfigurationRecord.created_at.desc()
                ).offset(offset).limit(limit).all()
                
                return [c.to_dict() for c in configurations], total
                
        except Exception as e:
            logger.error(f"Failed to list configurations: {e}")
            return [], 0
    
    def get_default_configuration(
        self,
        model_name: str,
        task_type: str,
        tenant_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """获取模型的默认配置"""
        try:
            if self._use_memory_storage:
                for cfg_data in self._configurations.values():
                    if (cfg_data.get('model_name') == model_name and
                        cfg_data.get('task_type') == task_type and
                        cfg_data.get('is_default') == True):
                        if tenant_id and cfg_data.get('tenant_id') != tenant_id:
                            continue
                        return cfg_data
                return None
            
            from backend.schemas.training_models import ModelConfigurationRecord
            
            with self._db_manager.get_db_session() as db:
                query = db.query(ModelConfigurationRecord).filter(
                    ModelConfigurationRecord.model_name == model_name,
                    ModelConfigurationRecord.task_type == task_type,
                    ModelConfigurationRecord.is_default == True
                )
                if tenant_id:
                    query = query.filter(ModelConfigurationRecord.tenant_id == tenant_id)
                
                configuration = query.first()
                return configuration.to_dict() if configuration else None
                
        except Exception as e:
            logger.error(f"Failed to get default configuration: {e}")
            return None
    
    def update(self, configuration_id: str, update_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """更新配置记录"""
        try:
            if self._use_memory_storage:
                for record_id, cfg_data in self._configurations.items():
                    if cfg_data.get('configuration_id') == configuration_id:
                        cfg_data.update(update_data)
                        cfg_data['updated_at'] = datetime.utcnow().isoformat()
                        return cfg_data
                return None
            
            from backend.schemas.training_models import ModelConfigurationRecord
            
            with self._db_manager.get_db_session() as db:
                configuration = db.query(ModelConfigurationRecord).filter(
                    ModelConfigurationRecord.configuration_id == configuration_id
                ).first()
                
                if not configuration:
                    return None
                
                for key, value in update_data.items():
                    if key in ('dataset_info', 'hyperparameters', 'training_config', 
                              'hardware_config', 'full_config', 'tags', 'metadata_') and isinstance(value, (dict, list)):
                        value = json.dumps(value)
                    if hasattr(configuration, key):
                        setattr(configuration, key, value)
                
                configuration.updated_at = datetime.utcnow()
                db.commit()
                db.refresh(configuration)
                return configuration.to_dict()
                
        except Exception as e:
            logger.error(f"Failed to update configuration: {e}")
            return None
    
    def increment_usage(self, configuration_id: str) -> bool:
        """增加配置使用计数"""
        try:
            if self._use_memory_storage:
                for cfg_data in self._configurations.values():
                    if cfg_data.get('configuration_id') == configuration_id:
                        cfg_data['usage_count'] = cfg_data.get('usage_count', 0) + 1
                        cfg_data['last_used_at'] = datetime.utcnow().isoformat()
                        return True
                return False
            
            from backend.schemas.training_models import ModelConfigurationRecord
            
            with self._db_manager.get_db_session() as db:
                configuration = db.query(ModelConfigurationRecord).filter(
                    ModelConfigurationRecord.configuration_id == configuration_id
                ).first()
                
                if not configuration:
                    return False
                
                configuration.usage_count = (configuration.usage_count or 0) + 1
                configuration.last_used_at = datetime.utcnow()
                db.commit()
                return True
                
        except Exception as e:
            logger.error(f"Failed to increment usage: {e}")
            return False
    
    def delete(self, configuration_id: str) -> bool:
        """删除配置记录"""
        try:
            if self._use_memory_storage:
                for record_id, cfg_data in list(self._configurations.items()):
                    if cfg_data.get('configuration_id') == configuration_id:
                        del self._configurations[record_id]
                        return True
                return False
            
            from backend.schemas.training_models import ModelConfigurationRecord
            
            with self._db_manager.get_db_session() as db:
                configuration = db.query(ModelConfigurationRecord).filter(
                    ModelConfigurationRecord.configuration_id == configuration_id
                ).first()
                
                if not configuration:
                    return False
                
                db.delete(configuration)
                db.commit()
                return True
                
        except Exception as e:
            logger.error(f"Failed to delete configuration: {e}")
            return False


class ModelCatalogRepository:
    """模型目录数据访问层"""
    
    def __init__(self, use_memory_storage: bool = False):
        """初始化模型目录仓库"""
        self._use_memory_storage = use_memory_storage
        
        if use_memory_storage:
            self._catalog_entries: Dict[str, Dict] = {}
            self._db_manager = None
        else:
            try:
                from backend.modules.database.manager import get_database_manager
                self._db_manager = get_database_manager()
            except ImportError:
                logger.warning("Cannot import database service, falling back to memory storage")
                self._use_memory_storage = True
                self._catalog_entries: Dict[str, Dict] = {}
                self._db_manager = None
    
    def create(self, entry_data: Dict[str, Any]) -> Dict[str, Any]:
        """创建目录条目"""
        try:
            record_id = entry_data.get('id') or _generate_id()
            
            if self._use_memory_storage:
                entry_data['id'] = record_id
                entry_data['created_at'] = datetime.utcnow().isoformat()
                entry_data['updated_at'] = datetime.utcnow().isoformat()
                entry_data.setdefault('is_enabled', True)
                self._catalog_entries[record_id] = entry_data
                return entry_data
            
            from backend.schemas.training_models import ModelCatalogEntry
            
            with self._db_manager.get_db_session() as db:
                entry = ModelCatalogEntry(
                    id=record_id,
                    tenant_id=entry_data.get('tenant_id'),
                    model_name=entry_data['model_name'],
                    task_type=entry_data['task_type'],
                    framework=entry_data['framework'],
                    model_type=entry_data['model_type'],
                    display_name=entry_data.get('display_name'),
                    description=entry_data.get('description'),
                    version=entry_data.get('version', '1.0.0'),
                    performance_metrics=json.dumps(entry_data.get('performance_metrics', {})) if entry_data.get('performance_metrics') else None,
                    benchmark_results=json.dumps(entry_data.get('benchmark_results', {})) if entry_data.get('benchmark_results') else None,
                    hardware_requirements=json.dumps(entry_data.get('hardware_requirements', {})) if entry_data.get('hardware_requirements') else None,
                    min_gpu_memory=entry_data.get('min_gpu_memory'),
                    min_cpu_cores=entry_data.get('min_cpu_cores'),
                    default_hyperparameters=json.dumps(entry_data.get('default_hyperparameters', {})) if entry_data.get('default_hyperparameters') else None,
                    default_training_config=json.dumps(entry_data.get('default_training_config', {})) if entry_data.get('default_training_config') else None,
                    tags=json.dumps(entry_data.get('tags', [])) if entry_data.get('tags') else None,
                    categories=json.dumps(entry_data.get('categories', [])) if entry_data.get('categories') else None,
                    recommended_for=json.dumps(entry_data.get('recommended_for', [])) if entry_data.get('recommended_for') else None,
                    is_enabled=entry_data.get('is_enabled', True),
                    is_public=entry_data.get('is_public', True),
                    is_system=entry_data.get('is_system', False),
                    sort_order=entry_data.get('sort_order', 0),
                    priority=entry_data.get('priority', 0),
                    source_url=entry_data.get('source_url'),
                    documentation_url=entry_data.get('documentation_url'),
                    metadata_=json.dumps(entry_data.get('metadata', {})) if entry_data.get('metadata') else None
                )
                db.add(entry)
                db.commit()
                db.refresh(entry)
                return entry.to_dict()
                
        except Exception as e:
            logger.error(f"Failed to create catalog entry: {e}")
            raise
    
    def get_by_model_name(
        self,
        model_name: str,
        task_type: Optional[str] = None,
        tenant_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """根据模型名称获取目录条目"""
        try:
            if self._use_memory_storage:
                for entry_data in self._catalog_entries.values():
                    if entry_data.get('model_name') == model_name:
                        if task_type and entry_data.get('task_type') != task_type:
                            continue
                        if tenant_id and entry_data.get('tenant_id') != tenant_id and not entry_data.get('is_system'):
                            continue
                        return entry_data
                return None
            
            from backend.schemas.training_models import ModelCatalogEntry
            
            with self._db_manager.get_db_session() as db:
                query = db.query(ModelCatalogEntry).filter(
                    ModelCatalogEntry.model_name == model_name,
                    ModelCatalogEntry.is_enabled == True
                )
                if task_type:
                    query = query.filter(ModelCatalogEntry.task_type == task_type)
                
                entry = query.first()
                return entry.to_dict() if entry else None
                
        except Exception as e:
            logger.error(f"Failed to get catalog entry: {e}")
            return None
    
    def list_by_task_type(
        self,
        task_type: str,
        tenant_id: Optional[str] = None,
        framework: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> Tuple[List[Dict[str, Any]], int]:
        """获取任务类型的模型列表"""
        try:
            if self._use_memory_storage:
                entries = [
                    e for e in self._catalog_entries.values()
                    if e.get('task_type') == task_type and e.get('is_enabled', True)
                ]
                if tenant_id:
                    entries = [e for e in entries if e.get('tenant_id') == tenant_id or e.get('is_system')]
                if framework:
                    entries = [e for e in entries if e.get('framework') == framework]
                
                entries.sort(key=lambda x: (x.get('priority', 0), x.get('sort_order', 0)), reverse=True)
                total = len(entries)
                return entries[offset:offset + limit], total
            
            from backend.schemas.training_models import ModelCatalogEntry
            
            with self._db_manager.get_db_session() as db:
                query = db.query(ModelCatalogEntry).filter(
                    ModelCatalogEntry.task_type == task_type,
                    ModelCatalogEntry.is_enabled == True
                )
                if framework:
                    query = query.filter(ModelCatalogEntry.framework == framework)
                
                total = query.count()
                entries = query.order_by(
                    ModelCatalogEntry.priority.desc(),
                    ModelCatalogEntry.sort_order
                ).offset(offset).limit(limit).all()
                
                return [e.to_dict() for e in entries], total
                
        except Exception as e:
            logger.error(f"Failed to list catalog entries: {e}")
            return [], 0
    
    def search(
        self,
        query: str,
        tenant_id: Optional[str] = None,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """搜索模型目录"""
        try:
            query_lower = query.lower()
            
            if self._use_memory_storage:
                results = []
                for entry_data in self._catalog_entries.values():
                    if not entry_data.get('is_enabled', True):
                        continue
                    
                    # 搜索模型名称、描述、标签
                    model_name = entry_data.get('model_name', '').lower()
                    description = entry_data.get('description', '').lower()
                    tags = entry_data.get('tags', [])
                    
                    if (query_lower in model_name or
                        query_lower in description or
                        any(query_lower in str(tag).lower() for tag in tags)):
                        results.append(entry_data)
                        
                        if len(results) >= limit:
                            break
                
                return results
            
            from backend.schemas.training_models import ModelCatalogEntry
            
            with self._db_manager.get_db_session() as db:
                entries = db.query(ModelCatalogEntry).filter(
                    ModelCatalogEntry.is_enabled == True,
                    (ModelCatalogEntry.model_name.ilike(f'%{query}%') |
                     ModelCatalogEntry.description.ilike(f'%{query}%'))
                ).limit(limit).all()
                
                return [e.to_dict() for e in entries]
                
        except Exception as e:
            logger.error(f"Failed to search catalog: {e}")
            return []
    
    def update(self, model_name: str, update_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """更新目录条目"""
        try:
            if self._use_memory_storage:
                for record_id, entry_data in self._catalog_entries.items():
                    if entry_data.get('model_name') == model_name:
                        entry_data.update(update_data)
                        entry_data['updated_at'] = datetime.utcnow().isoformat()
                        return entry_data
                return None
            
            from backend.schemas.training_models import ModelCatalogEntry
            
            with self._db_manager.get_db_session() as db:
                entry = db.query(ModelCatalogEntry).filter(
                    ModelCatalogEntry.model_name == model_name
                ).first()
                
                if not entry:
                    return None
                
                for key, value in update_data.items():
                    if key in ('performance_metrics', 'benchmark_results', 'hardware_requirements',
                              'default_hyperparameters', 'default_training_config', 'tags',
                              'categories', 'recommended_for', 'metadata_') and isinstance(value, (dict, list)):
                        value = json.dumps(value)
                    if hasattr(entry, key):
                        setattr(entry, key, value)
                
                entry.updated_at = datetime.utcnow()
                db.commit()
                db.refresh(entry)
                return entry.to_dict()
                
        except Exception as e:
            logger.error(f"Failed to update catalog entry: {e}")
            return None
    
    def increment_usage(self, model_name: str, increment_type: str = 'usage') -> bool:
        """增加模型使用计数"""
        try:
            if self._use_memory_storage:
                for entry_data in self._catalog_entries.values():
                    if entry_data.get('model_name') == model_name:
                        count_field = f'{increment_type}_count'
                        entry_data[count_field] = entry_data.get(count_field, 0) + 1
                        return True
                return False
            
            from backend.schemas.training_models import ModelCatalogEntry
            
            with self._db_manager.get_db_session() as db:
                entry = db.query(ModelCatalogEntry).filter(
                    ModelCatalogEntry.model_name == model_name
                ).first()
                
                if not entry:
                    return False
                
                if increment_type == 'usage':
                    entry.usage_count = (entry.usage_count or 0) + 1
                elif increment_type == 'recommendation':
                    entry.recommendation_count = (entry.recommendation_count or 0) + 1
                elif increment_type == 'selection':
                    entry.selection_count = (entry.selection_count or 0) + 1
                
                db.commit()
                return True
                
        except Exception as e:
            logger.error(f"Failed to increment usage: {e}")
            return False
    
    def get_task_types(self, tenant_id: Optional[str] = None) -> List[str]:
        """获取所有任务类型"""
        try:
            if self._use_memory_storage:
                task_types = set()
                for entry_data in self._catalog_entries.values():
                    if entry_data.get('is_enabled', True):
                        task_types.add(entry_data.get('task_type'))
                return list(task_types)
            
            from backend.schemas.training_models import ModelCatalogEntry
            
            with self._db_manager.get_db_session() as db:
                result = db.query(ModelCatalogEntry.task_type).filter(
                    ModelCatalogEntry.is_enabled == True
                ).distinct().all()
                
                return [r[0] for r in result if r[0]]
                
        except Exception as e:
            logger.error(f"Failed to get task types: {e}")
            return []


# 全局仓库实例
_recommendation_repository = None
_configuration_repository = None
_catalog_repository = None


def get_model_recommendation_repository(use_memory_storage: bool = False) -> ModelRecommendationRepository:
    """获取模型推荐仓库实例"""
    global _recommendation_repository
    if _recommendation_repository is None:
        _recommendation_repository = ModelRecommendationRepository(use_memory_storage=use_memory_storage)
    return _recommendation_repository


def get_model_configuration_repository(use_memory_storage: bool = False) -> ModelConfigurationRepository:
    """获取模型配置仓库实例"""
    global _configuration_repository
    if _configuration_repository is None:
        _configuration_repository = ModelConfigurationRepository(use_memory_storage=use_memory_storage)
    return _configuration_repository


def get_model_catalog_repository(use_memory_storage: bool = False) -> ModelCatalogRepository:
    """获取模型目录仓库实例"""
    global _catalog_repository
    if _catalog_repository is None:
        _catalog_repository = ModelCatalogRepository(use_memory_storage=use_memory_storage)
    return _catalog_repository


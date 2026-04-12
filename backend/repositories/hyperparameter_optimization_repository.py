"""超参数优化数据仓库

提供超参数优化相关的数据访问操作，支持内存存储和数据库持久化。
"""

import sys
import os
import logging
import uuid
import json
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime
from dataclasses import dataclass, field, asdict

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

logger = logging.getLogger(__name__)


# ==============================================================================
# 内存存储数据类
# ==============================================================================

@dataclass
class OptimizationData:
    """优化任务内存数据"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = ""
    tenant_id: str = ""
    name: str = ""
    description: str = ""
    scenario_type: str = ""
    optimization_method: str = "random"
    status: str = "pending"
    search_space: List[Dict] = field(default_factory=list)
    training_config: Dict = field(default_factory=dict)
    max_trials: int = 10
    current_trial: int = 0
    best_params: Dict = field(default_factory=dict)
    best_score: float = None
    best_trial_id: str = None
    total_trials: int = 0
    successful_trials: int = 0
    failed_trials: int = 0
    avg_trial_duration: float = None
    model_id: str = None
    dataset_id: str = None
    tags: List[str] = field(default_factory=list)
    extra_data: Dict = field(default_factory=dict)
    started_at: datetime = None
    completed_at: datetime = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


@dataclass
class TrialData:
    """试验记录内存数据"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    optimization_id: str = ""
    trial_number: int = 0
    status: str = "pending"
    params: Dict = field(default_factory=dict)
    score: float = None
    metrics: Dict = field(default_factory=dict)
    training_session_id: str = None
    training_loss: float = None
    validation_loss: float = None
    training_accuracy: float = None
    validation_accuracy: float = None
    started_at: datetime = None
    completed_at: datetime = None
    duration_seconds: float = None
    error_message: str = None
    error_details: Dict = None
    extra_data: Dict = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


@dataclass
class SearchSpaceData:
    """搜索空间模板内存数据"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = ""
    tenant_id: str = ""
    name: str = ""
    description: str = ""
    scenario_type: str = ""
    is_default: bool = False
    is_public: bool = False
    parameters: List[Dict] = field(default_factory=list)
    recommended_method: str = "random"
    recommended_trials: int = 10
    usage_count: int = 0
    tags: List[str] = field(default_factory=list)
    extra_data: Dict = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


# ==============================================================================
# 超参数优化任务仓库
# ==============================================================================

class HyperparameterOptimizationRepository:
    """超参数优化任务仓库"""
    
    def __init__(self, use_memory_storage: bool = False):
        """初始化仓库
        
        Args:
            use_memory_storage: 是否使用内存存储
        """
        self._use_memory_storage = use_memory_storage
        self._memory_storage: Dict[str, OptimizationData] = {}
        
        if not use_memory_storage:
            self._init_database()
    
    def _init_database(self):
        """初始化数据库连接"""
        try:
            from backend.core.database import get_db_session
            self._get_session = get_db_session
            logger.info("HyperparameterOptimizationRepository: Database connection initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize database, falling back to memory storage: {e}")
            self._use_memory_storage = True
    
    def create(self, data: Dict[str, Any]) -> Any:
        """创建优化任务
        
        Args:
            data: 优化任务数据
            
        Returns:
            创建的优化任务
        """
        if self._use_memory_storage:
            return self._create_memory(data)
        return self._create_db(data)
    
    def _create_memory(self, data: Dict[str, Any]) -> OptimizationData:
        """内存存储创建"""
        optimization = OptimizationData(
            id=data.get('id', str(uuid.uuid4())),
            user_id=data.get('user_id', ''),
            tenant_id=data.get('tenant_id', ''),
            name=data.get('name', ''),
            description=data.get('description', ''),
            scenario_type=data.get('scenario_type', ''),
            optimization_method=data.get('optimization_method', 'random'),
            status=data.get('status', 'pending'),
            search_space=data.get('search_space', []),
            training_config=data.get('training_config', {}),
            max_trials=data.get('max_trials', 10),
            current_trial=data.get('current_trial', 0),
            model_id=data.get('model_id'),
            dataset_id=data.get('dataset_id'),
            tags=data.get('tags', [])
        )
        self._memory_storage[optimization.id] = optimization
        logger.info(f"Created optimization in memory: {optimization.id}")
        return optimization
    
    def _create_db(self, data: Dict[str, Any]) -> Any:
        """数据库存储创建"""
        try:
            from backend.schemas.training_models import HyperparameterOptimization
            from backend.core.database import get_db_session
            
            with get_db_session() as session:
                optimization = HyperparameterOptimization(
                    user_id=data.get('user_id'),
                    tenant_id=data.get('tenant_id'),
                    name=data.get('name'),
                    description=data.get('description'),
                    scenario_type=data.get('scenario_type'),
                    optimization_method=data.get('optimization_method', 'random'),
                    status=data.get('status', 'pending'),
                    search_space=data.get('search_space'),
                    training_config=data.get('training_config'),
                    max_trials=data.get('max_trials', 10),
                    model_id=data.get('model_id'),
                    dataset_id=data.get('dataset_id'),
                    tags=data.get('tags')
                )
                session.add(optimization)
                session.commit()
                session.refresh(optimization)
                logger.info(f"Created optimization in database: {optimization.id}")
                return optimization
        except Exception as e:
            logger.error(f"Failed to create optimization in database: {e}")
            raise
    
    def get_by_id(self, optimization_id: str) -> Optional[Any]:
        """根据ID获取优化任务"""
        if self._use_memory_storage:
            return self._memory_storage.get(optimization_id)
        return self._get_by_id_db(optimization_id)
    
    def _get_by_id_db(self, optimization_id: str) -> Optional[Any]:
        """数据库获取"""
        try:
            from backend.schemas.training_models import HyperparameterOptimization
            from backend.core.database import get_db_session
            
            with get_db_session() as session:
                optimization = session.query(HyperparameterOptimization).filter(
                    HyperparameterOptimization.id == optimization_id
                ).first()
                return optimization
        except Exception as e:
            logger.error(f"Failed to get optimization from database: {e}")
            return None
    
    def update(self, optimization_id: str, data: Dict[str, Any]) -> Optional[Any]:
        """更新优化任务"""
        if self._use_memory_storage:
            return self._update_memory(optimization_id, data)
        return self._update_db(optimization_id, data)
    
    def _update_memory(self, optimization_id: str, data: Dict[str, Any]) -> Optional[OptimizationData]:
        """内存存储更新"""
        if optimization_id not in self._memory_storage:
            return None
        optimization = self._memory_storage[optimization_id]
        for key, value in data.items():
            if hasattr(optimization, key):
                setattr(optimization, key, value)
        optimization.updated_at = datetime.now()
        return optimization
    
    def _update_db(self, optimization_id: str, data: Dict[str, Any]) -> Optional[Any]:
        """数据库存储更新"""
        try:
            from backend.schemas.training_models import HyperparameterOptimization
            from backend.core.database import get_db_session
            
            with get_db_session() as session:
                optimization = session.query(HyperparameterOptimization).filter(
                    HyperparameterOptimization.id == optimization_id
                ).first()
                
                if not optimization:
                    return None
                
                for key, value in data.items():
                    if hasattr(optimization, key):
                        setattr(optimization, key, value)
                
                session.commit()
                session.refresh(optimization)
                return optimization
        except Exception as e:
            logger.error(f"Failed to update optimization in database: {e}")
            raise
    
    def delete(self, optimization_id: str) -> bool:
        """删除优化任务"""
        if self._use_memory_storage:
            if optimization_id in self._memory_storage:
                del self._memory_storage[optimization_id]
                return True
            return False
        return self._delete_db(optimization_id)
    
    def _delete_db(self, optimization_id: str) -> bool:
        """数据库删除"""
        try:
            from backend.schemas.training_models import HyperparameterOptimization
            from backend.core.database import get_db_session
            
            with get_db_session() as session:
                optimization = session.query(HyperparameterOptimization).filter(
                    HyperparameterOptimization.id == optimization_id
                ).first()
                
                if not optimization:
                    return False
                
                session.delete(optimization)
                session.commit()
                return True
        except Exception as e:
            logger.error(f"Failed to delete optimization from database: {e}")
            return False
    
    def list_by_tenant(self, tenant_id: str, user_id: Optional[str] = None,
                      status: Optional[str] = None, scenario_type: Optional[str] = None,
                      limit: int = 20, offset: int = 0) -> Tuple[List[Any], int]:
        """按租户列出优化任务"""
        if self._use_memory_storage:
            return self._list_by_tenant_memory(tenant_id, user_id, status, scenario_type, limit, offset)
        return self._list_by_tenant_db(tenant_id, user_id, status, scenario_type, limit, offset)
    
    def _list_by_tenant_memory(self, tenant_id: str, user_id: Optional[str],
                               status: Optional[str], scenario_type: Optional[str],
                               limit: int, offset: int) -> Tuple[List[OptimizationData], int]:
        """内存存储列表"""
        results = []
        for opt in self._memory_storage.values():
            if opt.tenant_id != tenant_id:
                continue
            if user_id and opt.user_id != user_id:
                continue
            if status and opt.status != status:
                continue
            if scenario_type and opt.scenario_type != scenario_type:
                continue
            results.append(opt)
        
        # 按创建时间排序
        results.sort(key=lambda x: x.created_at or datetime.min, reverse=True)
        total = len(results)
        return results[offset:offset + limit], total
    
    def _list_by_tenant_db(self, tenant_id: str, user_id: Optional[str],
                          status: Optional[str], scenario_type: Optional[str],
                          limit: int, offset: int) -> Tuple[List[Any], int]:
        """数据库存储列表"""
        try:
            from backend.schemas.training_models import HyperparameterOptimization
            from backend.core.database import get_db_session
            
            with get_db_session() as session:
                query = session.query(HyperparameterOptimization).filter(
                    HyperparameterOptimization.tenant_id == tenant_id
                )
                
                if user_id:
                    query = query.filter(HyperparameterOptimization.user_id == user_id)
                if status:
                    query = query.filter(HyperparameterOptimization.status == status)
                if scenario_type:
                    query = query.filter(HyperparameterOptimization.scenario_type == scenario_type)
                
                total = query.count()
                results = query.order_by(
                    HyperparameterOptimization.created_at.desc()
                ).offset(offset).limit(limit).all()
                
                return results, total
        except Exception as e:
            logger.error(f"Failed to list optimizations from database: {e}")
            return [], 0
    
    def get_statistics(self, tenant_id: str, user_id: Optional[str] = None) -> Dict[str, Any]:
        """获取优化统计信息"""
        if self._use_memory_storage:
            return self._get_statistics_memory(tenant_id, user_id)
        return self._get_statistics_db(tenant_id, user_id)
    
    def _get_statistics_memory(self, tenant_id: str, user_id: Optional[str]) -> Dict[str, Any]:
        """内存统计"""
        stats = {
            'total': 0,
            'by_status': {'pending': 0, 'running': 0, 'completed': 0, 'failed': 0, 'cancelled': 0},
            'by_method': {'random': 0, 'grid': 0, 'bayesian': 0},
            'avg_trials': 0,
            'avg_best_score': 0
        }
        
        total_trials = 0
        total_score = 0
        score_count = 0
        
        for opt in self._memory_storage.values():
            if opt.tenant_id != tenant_id:
                continue
            if user_id and opt.user_id != user_id:
                continue
            
            stats['total'] += 1
            if opt.status in stats['by_status']:
                stats['by_status'][opt.status] += 1
            if opt.optimization_method in stats['by_method']:
                stats['by_method'][opt.optimization_method] += 1
            
            total_trials += opt.total_trials
            if opt.best_score is not None:
                total_score += opt.best_score
                score_count += 1
        
        if stats['total'] > 0:
            stats['avg_trials'] = total_trials / stats['total']
        if score_count > 0:
            stats['avg_best_score'] = total_score / score_count
        
        return stats
    
    def _get_statistics_db(self, tenant_id: str, user_id: Optional[str]) -> Dict[str, Any]:
        """数据库统计"""
        try:
            from backend.schemas.training_models import HyperparameterOptimization
            from backend.core.database import get_db_session
            from sqlalchemy.sql.functions import count, avg
            
            with get_db_session() as session:
                query = session.query(HyperparameterOptimization).filter(
                    HyperparameterOptimization.tenant_id == tenant_id
                )
                if user_id:
                    query = query.filter(HyperparameterOptimization.user_id == user_id)
                
                total = query.count()
                
                # 按状态统计
                status_stats = session.query(
                    HyperparameterOptimization.status,
                    count(HyperparameterOptimization.id).label('count')
                ).filter(
                    HyperparameterOptimization.tenant_id == tenant_id
                ).group_by(HyperparameterOptimization.status).all()
                
                by_status = {s: c for s, c in status_stats}
                
                # 按方法统计
                method_stats = session.query(
                    HyperparameterOptimization.optimization_method,
                    count(HyperparameterOptimization.id)
                ).filter(
                    HyperparameterOptimization.tenant_id == tenant_id
                ).group_by(HyperparameterOptimization.optimization_method).all()
                
                by_method = {m: c for m, c in method_stats}
                
                # 平均值
                avg_stats = session.query(
                    avg(HyperparameterOptimization.total_trials),
                    avg(HyperparameterOptimization.best_score)
                ).filter(
                    HyperparameterOptimization.tenant_id == tenant_id
                ).first()
                
                return {
                    'total': total,
                    'by_status': by_status,
                    'by_method': by_method,
                    'avg_trials': float(avg_stats[0] or 0),
                    'avg_best_score': float(avg_stats[1] or 0)
                }
        except Exception as e:
            logger.error(f"Failed to get statistics from database: {e}")
            return {'total': 0, 'by_status': {}, 'by_method': {}, 'avg_trials': 0, 'avg_best_score': 0}


# ==============================================================================
# 超参数试验仓库
# ==============================================================================

class HyperparameterTrialRepository:
    """超参数试验仓库"""
    
    def __init__(self, use_memory_storage: bool = False):
        """初始化仓库"""
        self._use_memory_storage = use_memory_storage
        self._memory_storage: Dict[str, TrialData] = {}
        
        if not use_memory_storage:
            self._init_database()
    
    def _init_database(self):
        """初始化数据库连接"""
        try:
            from backend.core.database import get_db_session
            self._get_session = get_db_session
            logger.info("HyperparameterTrialRepository: Database connection initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize database, falling back to memory storage: {e}")
            self._use_memory_storage = True
    
    def create(self, data: Dict[str, Any]) -> Any:
        """创建试验记录"""
        if self._use_memory_storage:
            return self._create_memory(data)
        return self._create_db(data)
    
    def _create_memory(self, data: Dict[str, Any]) -> TrialData:
        """内存存储创建"""
        trial = TrialData(
            id=data.get('id', str(uuid.uuid4())),
            optimization_id=data.get('optimization_id', ''),
            trial_number=data.get('trial_number', 0),
            status=data.get('status', 'pending'),
            params=data.get('params', {}),
            score=data.get('score'),
            metrics=data.get('metrics', {}),
            training_session_id=data.get('training_session_id'),
            started_at=data.get('started_at')
        )
        self._memory_storage[trial.id] = trial
        return trial
    
    def _create_db(self, data: Dict[str, Any]) -> Any:
        """数据库存储创建"""
        try:
            from backend.schemas.training_models import HyperparameterTrial
            from backend.core.database import get_db_session
            
            with get_db_session() as session:
                trial = HyperparameterTrial(
                    optimization_id=data.get('optimization_id'),
                    trial_number=data.get('trial_number'),
                    status=data.get('status', 'pending'),
                    params=data.get('params'),
                    started_at=data.get('started_at')
                )
                session.add(trial)
                session.commit()
                session.refresh(trial)
                return trial
        except Exception as e:
            logger.error(f"Failed to create trial in database: {e}")
            raise
    
    def get_by_id(self, trial_id: str) -> Optional[Any]:
        """根据ID获取试验"""
        if self._use_memory_storage:
            return self._memory_storage.get(trial_id)
        return self._get_by_id_db(trial_id)
    
    def _get_by_id_db(self, trial_id: str) -> Optional[Any]:
        """数据库获取"""
        try:
            from backend.schemas.training_models import HyperparameterTrial
            from backend.core.database import get_db_session
            
            with get_db_session() as session:
                return session.query(HyperparameterTrial).filter(
                    HyperparameterTrial.id == trial_id
                ).first()
        except Exception as e:
            logger.error(f"Failed to get trial from database: {e}")
            return None
    
    def update(self, trial_id: str, data: Dict[str, Any]) -> Optional[Any]:
        """更新试验记录"""
        if self._use_memory_storage:
            return self._update_memory(trial_id, data)
        return self._update_db(trial_id, data)
    
    def _update_memory(self, trial_id: str, data: Dict[str, Any]) -> Optional[TrialData]:
        """内存存储更新"""
        if trial_id not in self._memory_storage:
            return None
        trial = self._memory_storage[trial_id]
        for key, value in data.items():
            if hasattr(trial, key):
                setattr(trial, key, value)
        trial.updated_at = datetime.now()
        return trial
    
    def _update_db(self, trial_id: str, data: Dict[str, Any]) -> Optional[Any]:
        """数据库存储更新"""
        try:
            from backend.schemas.training_models import HyperparameterTrial
            from backend.core.database import get_db_session
            
            with get_db_session() as session:
                trial = session.query(HyperparameterTrial).filter(
                    HyperparameterTrial.id == trial_id
                ).first()
                
                if not trial:
                    return None
                
                for key, value in data.items():
                    if hasattr(trial, key):
                        setattr(trial, key, value)
                
                session.commit()
                session.refresh(trial)
                return trial
        except Exception as e:
            logger.error(f"Failed to update trial in database: {e}")
            raise
    
    def get_by_optimization(self, optimization_id: str, 
                           status: Optional[str] = None) -> List[Any]:
        """获取优化任务的所有试验"""
        if self._use_memory_storage:
            return self._get_by_optimization_memory(optimization_id, status)
        return self._get_by_optimization_db(optimization_id, status)
    
    def _get_by_optimization_memory(self, optimization_id: str, 
                                    status: Optional[str]) -> List[TrialData]:
        """内存获取"""
        results = []
        for trial in self._memory_storage.values():
            if trial.optimization_id != optimization_id:
                continue
            if status and trial.status != status:
                continue
            results.append(trial)
        results.sort(key=lambda x: x.trial_number)
        return results
    
    def _get_by_optimization_db(self, optimization_id: str, 
                               status: Optional[str]) -> List[Any]:
        """数据库获取"""
        try:
            from backend.schemas.training_models import HyperparameterTrial
            from backend.core.database import get_db_session
            
            with get_db_session() as session:
                query = session.query(HyperparameterTrial).filter(
                    HyperparameterTrial.optimization_id == optimization_id
                )
                if status:
                    query = query.filter(HyperparameterTrial.status == status)
                
                return query.order_by(HyperparameterTrial.trial_number).all()
        except Exception as e:
            logger.error(f"Failed to get trials from database: {e}")
            return []
    
    def get_best_trial(self, optimization_id: str) -> Optional[Any]:
        """获取最佳试验"""
        if self._use_memory_storage:
            return self._get_best_trial_memory(optimization_id)
        return self._get_best_trial_db(optimization_id)
    
    def _get_best_trial_memory(self, optimization_id: str) -> Optional[TrialData]:
        """内存获取最佳试验"""
        best_trial = None
        best_score = float('-inf')
        
        for trial in self._memory_storage.values():
            if trial.optimization_id != optimization_id:
                continue
            if trial.status != 'completed':
                continue
            if trial.score is not None and trial.score > best_score:
                best_score = trial.score
                best_trial = trial
        
        return best_trial
    
    def _get_best_trial_db(self, optimization_id: str) -> Optional[Any]:
        """数据库获取最佳试验"""
        try:
            from backend.schemas.training_models import HyperparameterTrial
            from backend.core.database import get_db_session
            
            with get_db_session() as session:
                return session.query(HyperparameterTrial).filter(
                    HyperparameterTrial.optimization_id == optimization_id,
                    HyperparameterTrial.status == 'completed'
                ).order_by(HyperparameterTrial.score.desc()).first()
        except Exception as e:
            logger.error(f"Failed to get best trial from database: {e}")
            return None


# ==============================================================================
# 搜索空间模板仓库
# ==============================================================================

class HyperparameterSearchSpaceRepository:
    """搜索空间模板仓库"""
    
    def __init__(self, use_memory_storage: bool = False):
        """初始化仓库"""
        self._use_memory_storage = use_memory_storage
        self._memory_storage: Dict[str, SearchSpaceData] = {}
        
        if not use_memory_storage:
            self._init_database()
        else:
            self._init_default_templates()
    
    def _init_database(self):
        """初始化数据库连接"""
        try:
            from backend.core.database import get_db_session
            self._get_session = get_db_session
            logger.info("HyperparameterSearchSpaceRepository: Database connection initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize database, falling back to memory storage: {e}")
            self._use_memory_storage = True
            self._init_default_templates()
    
    def _init_default_templates(self):
        """初始化默认模板"""
        default_templates = [
            SearchSpaceData(
                id='tpl_classification',
                name='分类任务标准搜索空间',
                description='适用于分类任务的常用超参数搜索空间',
                scenario_type='classification',
                is_default=True,
                is_public=True,
                parameters=[
                    {'name': 'learning_rate', 'type': 'float', 'low': 0.0001, 'high': 0.1},
                    {'name': 'batch_size', 'type': 'categorical', 'choices': [16, 32, 64, 128]},
                    {'name': 'epochs', 'type': 'int', 'low': 5, 'high': 50},
                    {'name': 'dropout_rate', 'type': 'float', 'low': 0.1, 'high': 0.5},
                    {'name': 'hidden_units', 'type': 'categorical', 'choices': [64, 128, 256, 512]}
                ],
                recommended_method='bayesian',
                recommended_trials=20
            ),
            SearchSpaceData(
                id='tpl_regression',
                name='回归任务标准搜索空间',
                description='适用于回归任务的常用超参数搜索空间',
                scenario_type='regression',
                is_default=True,
                is_public=True,
                parameters=[
                    {'name': 'learning_rate', 'type': 'float', 'low': 0.0001, 'high': 0.01},
                    {'name': 'batch_size', 'type': 'categorical', 'choices': [32, 64, 128]},
                    {'name': 'epochs', 'type': 'int', 'low': 10, 'high': 100},
                    {'name': 'regularization', 'type': 'float', 'low': 0.001, 'high': 0.1}
                ],
                recommended_method='bayesian',
                recommended_trials=15
            ),
            SearchSpaceData(
                id='tpl_nlp',
                name='NLP任务搜索空间',
                description='适用于自然语言处理任务的超参数搜索空间',
                scenario_type='nlp',
                is_default=True,
                is_public=True,
                parameters=[
                    {'name': 'learning_rate', 'type': 'float', 'low': 1e-5, 'high': 5e-5},
                    {'name': 'batch_size', 'type': 'categorical', 'choices': [8, 16, 32]},
                    {'name': 'max_length', 'type': 'categorical', 'choices': [128, 256, 512]},
                    {'name': 'warmup_steps', 'type': 'int', 'low': 100, 'high': 1000},
                    {'name': 'weight_decay', 'type': 'float', 'low': 0.0, 'high': 0.1}
                ],
                recommended_method='bayesian',
                recommended_trials=10
            ),
            SearchSpaceData(
                id='tpl_cv',
                name='计算机视觉任务搜索空间',
                description='适用于图像分类、目标检测等任务的超参数搜索空间',
                scenario_type='computer_vision',
                is_default=True,
                is_public=True,
                parameters=[
                    {'name': 'learning_rate', 'type': 'float', 'low': 0.001, 'high': 0.1},
                    {'name': 'batch_size', 'type': 'categorical', 'choices': [16, 32, 64]},
                    {'name': 'epochs', 'type': 'int', 'low': 10, 'high': 100},
                    {'name': 'augmentation_strength', 'type': 'float', 'low': 0.1, 'high': 0.5},
                    {'name': 'optimizer', 'type': 'categorical', 'choices': ['adam', 'sgd', 'adamw']}
                ],
                recommended_method='random',
                recommended_trials=25
            )
        ]
        
        for template in default_templates:
            self._memory_storage[template.id] = template
    
    def create(self, data: Dict[str, Any]) -> Any:
        """创建搜索空间模板"""
        if self._use_memory_storage:
            return self._create_memory(data)
        return self._create_db(data)
    
    def _create_memory(self, data: Dict[str, Any]) -> SearchSpaceData:
        """内存存储创建"""
        template = SearchSpaceData(
            id=data.get('id', str(uuid.uuid4())),
            user_id=data.get('user_id', ''),
            tenant_id=data.get('tenant_id', ''),
            name=data.get('name', ''),
            description=data.get('description', ''),
            scenario_type=data.get('scenario_type', ''),
            is_default=data.get('is_default', False),
            is_public=data.get('is_public', False),
            parameters=data.get('parameters', []),
            recommended_method=data.get('recommended_method', 'random'),
            recommended_trials=data.get('recommended_trials', 10),
            tags=data.get('tags', [])
        )
        self._memory_storage[template.id] = template
        return template
    
    def _create_db(self, data: Dict[str, Any]) -> Any:
        """数据库存储创建"""
        try:
            from backend.schemas.training_models import HyperparameterSearchSpace
            from backend.core.database import get_db_session
            
            with get_db_session() as session:
                template = HyperparameterSearchSpace(
                    user_id=data.get('user_id'),
                    tenant_id=data.get('tenant_id'),
                    name=data.get('name'),
                    description=data.get('description'),
                    scenario_type=data.get('scenario_type'),
                    is_default=data.get('is_default', False),
                    is_public=data.get('is_public', False),
                    parameters=data.get('parameters'),
                    recommended_method=data.get('recommended_method', 'random'),
                    recommended_trials=data.get('recommended_trials', 10),
                    tags=data.get('tags')
                )
                session.add(template)
                session.commit()
                session.refresh(template)
                return template
        except Exception as e:
            logger.error(f"Failed to create search space template in database: {e}")
            raise
    
    def get_by_id(self, template_id: str) -> Optional[Any]:
        """根据ID获取模板"""
        if self._use_memory_storage:
            return self._memory_storage.get(template_id)
        return self._get_by_id_db(template_id)
    
    def _get_by_id_db(self, template_id: str) -> Optional[Any]:
        """数据库获取"""
        try:
            from backend.schemas.training_models import HyperparameterSearchSpace
            from backend.core.database import get_db_session
            
            with get_db_session() as session:
                return session.query(HyperparameterSearchSpace).filter(
                    HyperparameterSearchSpace.id == template_id
                ).first()
        except Exception as e:
            logger.error(f"Failed to get search space template from database: {e}")
            return None
    
    def list_available(self, tenant_id: str, scenario_type: Optional[str] = None,
                      limit: int = 20, offset: int = 0) -> Tuple[List[Any], int]:
        """列出可用的模板（包括公开模板和租户自己的模板）"""
        if self._use_memory_storage:
            return self._list_available_memory(tenant_id, scenario_type, limit, offset)
        return self._list_available_db(tenant_id, scenario_type, limit, offset)
    
    def _list_available_memory(self, tenant_id: str, scenario_type: Optional[str],
                              limit: int, offset: int) -> Tuple[List[SearchSpaceData], int]:
        """内存列表"""
        results = []
        for template in self._memory_storage.values():
            if not (template.is_public or template.tenant_id == tenant_id):
                continue
            if scenario_type and template.scenario_type != scenario_type:
                continue
            results.append(template)
        
        results.sort(key=lambda x: (not x.is_default, x.name))
        total = len(results)
        return results[offset:offset + limit], total
    
    def _list_available_db(self, tenant_id: str, scenario_type: Optional[str],
                          limit: int, offset: int) -> Tuple[List[Any], int]:
        """数据库列表"""
        try:
            from backend.schemas.training_models import HyperparameterSearchSpace
            from backend.core.database import get_db_session
            from sqlalchemy import or_
            
            with get_db_session() as session:
                query = session.query(HyperparameterSearchSpace).filter(
                    or_(
                        HyperparameterSearchSpace.is_public == True,
                        HyperparameterSearchSpace.tenant_id == tenant_id
                    )
                )
                
                if scenario_type:
                    query = query.filter(HyperparameterSearchSpace.scenario_type == scenario_type)
                
                total = query.count()
                results = query.order_by(
                    HyperparameterSearchSpace.is_default.desc(),
                    HyperparameterSearchSpace.name
                ).offset(offset).limit(limit).all()
                
                return results, total
        except Exception as e:
            logger.error(f"Failed to list search space templates from database: {e}")
            return [], 0
    
    def increment_usage(self, template_id: str) -> bool:
        """增加使用计数"""
        if self._use_memory_storage:
            if template_id in self._memory_storage:
                self._memory_storage[template_id].usage_count += 1
                return True
            return False
        
        try:
            from backend.schemas.training_models import HyperparameterSearchSpace
            from backend.core.database import get_db_session
            
            with get_db_session() as session:
                template = session.query(HyperparameterSearchSpace).filter(
                    HyperparameterSearchSpace.id == template_id
                ).first()
                
                if template:
                    template.usage_count = (template.usage_count or 0) + 1
                    session.commit()
                    return True
                return False
        except Exception as e:
            logger.error(f"Failed to increment usage count: {e}")
            return False


# ==============================================================================
# 仓库工厂函数
# ==============================================================================

_optimization_repo_instance = None
_trial_repo_instance = None
_search_space_repo_instance = None


def get_hyperparameter_optimization_repository(use_memory_storage: bool = False) -> HyperparameterOptimizationRepository:
    """获取超参数优化仓库实例"""
    global _optimization_repo_instance
    if _optimization_repo_instance is None:
        _optimization_repo_instance = HyperparameterOptimizationRepository(use_memory_storage)
    return _optimization_repo_instance


def get_hyperparameter_trial_repository(use_memory_storage: bool = False) -> HyperparameterTrialRepository:
    """获取超参数试验仓库实例"""
    global _trial_repo_instance
    if _trial_repo_instance is None:
        _trial_repo_instance = HyperparameterTrialRepository(use_memory_storage)
    return _trial_repo_instance


def get_hyperparameter_search_space_repository(use_memory_storage: bool = False) -> HyperparameterSearchSpaceRepository:
    """获取搜索空间模板仓库实例"""
    global _search_space_repo_instance
    if _search_space_repo_instance is None:
        _search_space_repo_instance = HyperparameterSearchSpaceRepository(use_memory_storage)
    return _search_space_repo_instance



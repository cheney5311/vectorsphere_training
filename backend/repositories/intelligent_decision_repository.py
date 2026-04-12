"""智能决策数据仓库

提供智能决策相关的数据访问操作，支持内存存储和数据库持久化。
"""

import sys
import os
import logging
import uuid
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime
from dataclasses import dataclass, field, asdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

logger = logging.getLogger(__name__)


# ==============================================================================
# 内存存储数据类
# ==============================================================================

@dataclass
class DecisionData:
    """决策记录内存数据"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = ""
    tenant_id: str = ""
    decision_id: str = ""
    scenario: str = ""
    algorithm: str = ""
    status: str = "completed"
    inputs: Dict = field(default_factory=dict)
    constraints: Dict = field(default_factory=dict)
    history_context: List = field(default_factory=list)
    recommended_action: Dict = field(default_factory=dict)
    confidence: float = 0.0
    reasoning: str = ""
    alternatives: List = field(default_factory=list)
    execution_plan: Dict = field(default_factory=dict)
    executed: bool = False
    execution_result: Dict = None
    feedback_score: float = None
    feedback_comment: str = None
    tags: List[str] = field(default_factory=list)
    extra_data: Dict = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


@dataclass
class AdaptiveOptimizationData:
    """自适应优化内存数据"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = ""
    tenant_id: str = ""
    optimization_id: str = ""
    parameter_name: str = ""
    adjustment_strategy: str = ""
    adjustment_range: Dict = field(default_factory=dict)
    monitoring_metrics: List = field(default_factory=list)
    original_value: Any = None
    optimized_value: Any = None
    improvement_metric: str = ""
    improvement_value: float = 0.0
    adjustment_reason: str = ""
    applied: bool = False
    applied_at: datetime = None
    rollback: bool = False
    rollback_reason: str = None
    tags: List[str] = field(default_factory=list)
    extra_data: Dict = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


@dataclass
class KnowledgeData:
    """知识库内存数据"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = ""
    tenant_id: str = ""
    knowledge_type: str = ""
    category: str = ""
    title: str = ""
    content: Dict = field(default_factory=dict)
    related_entities: List = field(default_factory=list)
    relationships: List = field(default_factory=list)
    source: str = ""
    confidence: float = 1.0
    verified: bool = False
    usage_count: int = 0
    last_used_at: datetime = None
    is_active: bool = True
    is_public: bool = False
    tags: List[str] = field(default_factory=list)
    extra_data: Dict = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


@dataclass
class ExperienceData:
    """经验记录内存数据"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = ""
    tenant_id: str = ""
    experience_type: str = ""
    scenario: str = ""
    context: Dict = field(default_factory=dict)
    action: Dict = field(default_factory=dict)
    result: Dict = field(default_factory=dict)
    reward: float = 0.0
    decision_id: str = None
    effectiveness: float = None
    lessons_learned: str = ""
    is_positive: bool = True
    is_verified: bool = False
    tags: List[str] = field(default_factory=list)
    extra_data: Dict = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


# ==============================================================================
# 智能决策仓库
# ==============================================================================

class IntelligentDecisionRepository:
    """智能决策仓库"""
    
    def __init__(self, use_memory_storage: bool = False):
        self._use_memory_storage = use_memory_storage
        self._memory_storage: Dict[str, DecisionData] = {}
        
        if not use_memory_storage:
            self._init_database()
    
    def _init_database(self):
        """初始化数据库连接"""
        try:
            from backend.core.database import get_db_session
            self._get_session = get_db_session
            logger.info("IntelligentDecisionRepository: Database initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize database: {e}")
            self._use_memory_storage = True
    
    def create(self, data: Dict[str, Any]) -> Any:
        """创建决策记录"""
        if self._use_memory_storage:
            return self._create_memory(data)
        return self._create_db(data)
    
    def _create_memory(self, data: Dict[str, Any]) -> DecisionData:
        """内存存储创建"""
        decision = DecisionData(
            id=data.get('id', str(uuid.uuid4())),
            user_id=data.get('user_id', ''),
            tenant_id=data.get('tenant_id', ''),
            decision_id=data.get('decision_id', ''),
            scenario=data.get('scenario', ''),
            algorithm=data.get('algorithm', ''),
            status=data.get('status', 'completed'),
            inputs=data.get('inputs', {}),
            constraints=data.get('constraints', {}),
            history_context=data.get('history_context', []),
            recommended_action=data.get('recommended_action', {}),
            confidence=data.get('confidence', 0.0),
            reasoning=data.get('reasoning', ''),
            alternatives=data.get('alternatives', []),
            execution_plan=data.get('execution_plan', {}),
            tags=data.get('tags', [])
        )
        self._memory_storage[decision.id] = decision
        logger.info(f"Created decision in memory: {decision.decision_id}")
        return decision
    
    def _create_db(self, data: Dict[str, Any]) -> Any:
        """数据库存储创建"""
        try:
            from backend.schemas.training_models import IntelligentDecision
            from backend.core.database import get_db_session
            
            with get_db_session() as session:
                decision = IntelligentDecision(
                    user_id=data.get('user_id'),
                    tenant_id=data.get('tenant_id'),
                    decision_id=data.get('decision_id'),
                    scenario=data.get('scenario'),
                    algorithm=data.get('algorithm'),
                    status=data.get('status', 'completed'),
                    inputs=data.get('inputs'),
                    constraints=data.get('constraints'),
                    history_context=data.get('history_context'),
                    recommended_action=data.get('recommended_action'),
                    confidence=data.get('confidence'),
                    reasoning=data.get('reasoning'),
                    alternatives=data.get('alternatives'),
                    execution_plan=data.get('execution_plan'),
                    tags=data.get('tags')
                )
                session.add(decision)
                session.commit()
                session.refresh(decision)
                return decision
        except Exception as e:
            logger.error(f"Failed to create decision in database: {e}")
            raise
    
    def get_by_id(self, decision_id: str) -> Optional[Any]:
        """根据ID获取决策记录"""
        if self._use_memory_storage:
            return self._memory_storage.get(decision_id)
        return self._get_by_id_db(decision_id)
    
    def _get_by_id_db(self, decision_id: str) -> Optional[Any]:
        """数据库获取"""
        try:
            from backend.schemas.training_models import IntelligentDecision
            from backend.core.database import get_db_session
            
            with get_db_session() as session:
                return session.query(IntelligentDecision).filter(
                    IntelligentDecision.id == decision_id
                ).first()
        except Exception as e:
            logger.error(f"Failed to get decision from database: {e}")
            return None
    
    def get_by_decision_id(self, decision_id: str) -> Optional[Any]:
        """根据决策ID获取记录"""
        if self._use_memory_storage:
            for d in self._memory_storage.values():
                if d.decision_id == decision_id:
                    return d
            return None
        
        try:
            from backend.schemas.training_models import IntelligentDecision
            from backend.core.database import get_db_session
            
            with get_db_session() as session:
                return session.query(IntelligentDecision).filter(
                    IntelligentDecision.decision_id == decision_id
                ).first()
        except Exception as e:
            logger.error(f"Failed to get decision by decision_id: {e}")
            return None
    
    def update(self, id: str, data: Dict[str, Any]) -> Optional[Any]:
        """更新决策记录"""
        if self._use_memory_storage:
            return self._update_memory(id, data)
        return self._update_db(id, data)
    
    def _update_memory(self, id: str, data: Dict[str, Any]) -> Optional[DecisionData]:
        """内存存储更新"""
        if id not in self._memory_storage:
            return None
        decision = self._memory_storage[id]
        for key, value in data.items():
            if hasattr(decision, key):
                setattr(decision, key, value)
        decision.updated_at = datetime.now()
        return decision
    
    def _update_db(self, id: str, data: Dict[str, Any]) -> Optional[Any]:
        """数据库存储更新"""
        try:
            from backend.schemas.training_models import IntelligentDecision
            from backend.core.database import get_db_session
            
            with get_db_session() as session:
                decision = session.query(IntelligentDecision).filter(
                    IntelligentDecision.id == id
                ).first()
                
                if not decision:
                    return None
                
                for key, value in data.items():
                    if hasattr(decision, key):
                        setattr(decision, key, value)
                
                session.commit()
                session.refresh(decision)
                return decision
        except Exception as e:
            logger.error(f"Failed to update decision in database: {e}")
            raise
    
    def delete(self, id: str) -> bool:
        """删除决策记录"""
        if self._use_memory_storage:
            if id in self._memory_storage:
                del self._memory_storage[id]
                return True
            return False
        
        try:
            from backend.schemas.training_models import IntelligentDecision
            from backend.core.database import get_db_session
            
            with get_db_session() as session:
                decision = session.query(IntelligentDecision).filter(
                    IntelligentDecision.id == id
                ).first()
                
                if not decision:
                    return False
                
                session.delete(decision)
                session.commit()
                return True
        except Exception as e:
            logger.error(f"Failed to delete decision: {e}")
            return False
    
    def list_by_tenant(self, tenant_id: str, user_id: Optional[str] = None,
                      scenario: Optional[str] = None,
                      limit: int = 20, offset: int = 0) -> Tuple[List[Any], int]:
        """按租户列出决策记录"""
        if self._use_memory_storage:
            return self._list_by_tenant_memory(tenant_id, user_id, scenario, limit, offset)
        return self._list_by_tenant_db(tenant_id, user_id, scenario, limit, offset)
    
    def _list_by_tenant_memory(self, tenant_id: str, user_id: Optional[str],
                               scenario: Optional[str],
                               limit: int, offset: int) -> Tuple[List[DecisionData], int]:
        """内存存储列表"""
        results = []
        for d in self._memory_storage.values():
            if d.tenant_id != tenant_id:
                continue
            if user_id and d.user_id != user_id:
                continue
            if scenario and d.scenario != scenario:
                continue
            results.append(d)
        
        results.sort(key=lambda x: x.created_at or datetime.min, reverse=True)
        total = len(results)
        return results[offset:offset + limit], total
    
    def _list_by_tenant_db(self, tenant_id: str, user_id: Optional[str],
                          scenario: Optional[str],
                          limit: int, offset: int) -> Tuple[List[Any], int]:
        """数据库存储列表"""
        try:
            from backend.schemas.training_models import IntelligentDecision
            from backend.core.database import get_db_session
            
            with get_db_session() as session:
                query = session.query(IntelligentDecision).filter(
                    IntelligentDecision.tenant_id == tenant_id
                )
                
                if user_id:
                    query = query.filter(IntelligentDecision.user_id == user_id)
                if scenario:
                    query = query.filter(IntelligentDecision.scenario == scenario)
                
                total = query.count()
                results = query.order_by(
                    IntelligentDecision.created_at.desc()
                ).offset(offset).limit(limit).all()
                
                return results, total
        except Exception as e:
            logger.error(f"Failed to list decisions: {e}")
            return [], 0
    
    def get_statistics(self, tenant_id: str, user_id: Optional[str] = None) -> Dict[str, Any]:
        """获取决策统计信息"""
        if self._use_memory_storage:
            return self._get_statistics_memory(tenant_id, user_id)
        return self._get_statistics_db(tenant_id, user_id)
    
    def _get_statistics_memory(self, tenant_id: str, user_id: Optional[str]) -> Dict[str, Any]:
        """内存统计"""
        stats = {
            'total': 0,
            'by_scenario': {},
            'by_algorithm': {},
            'avg_confidence': 0,
            'executed_count': 0,
            'avg_feedback_score': 0
        }
        
        total_confidence = 0
        total_feedback = 0
        feedback_count = 0
        
        for d in self._memory_storage.values():
            if d.tenant_id != tenant_id:
                continue
            if user_id and d.user_id != user_id:
                continue
            
            stats['total'] += 1
            stats['by_scenario'][d.scenario] = stats['by_scenario'].get(d.scenario, 0) + 1
            stats['by_algorithm'][d.algorithm] = stats['by_algorithm'].get(d.algorithm, 0) + 1
            total_confidence += d.confidence
            
            if d.executed:
                stats['executed_count'] += 1
            if d.feedback_score is not None:
                total_feedback += d.feedback_score
                feedback_count += 1
        
        if stats['total'] > 0:
            stats['avg_confidence'] = total_confidence / stats['total']
        if feedback_count > 0:
            stats['avg_feedback_score'] = total_feedback / feedback_count
        
        return stats
    
    def _get_statistics_db(self, tenant_id: str, user_id: Optional[str]) -> Dict[str, Any]:
        """数据库统计"""
        try:
            from backend.schemas.training_models import IntelligentDecision
            from backend.core.database import get_db_session
            from sqlalchemy import func
            
            with get_db_session() as session:
                query = session.query(IntelligentDecision).filter(
                    IntelligentDecision.tenant_id == tenant_id
                )
                if user_id:
                    query = query.filter(IntelligentDecision.user_id == user_id)
                
                total = query.count()
                
                # 按场景统计
                scenario_stats = session.query(
                    IntelligentDecision.scenario,
                    func.count(IntelligentDecision.id)
                ).filter(
                    IntelligentDecision.tenant_id == tenant_id
                ).group_by(IntelligentDecision.scenario).all()
                
                # 按算法统计
                algo_stats = session.query(
                    IntelligentDecision.algorithm,
                    func.count(IntelligentDecision.id)
                ).filter(
                    IntelligentDecision.tenant_id == tenant_id
                ).group_by(IntelligentDecision.algorithm).all()
                
                # 平均值
                avg_stats = session.query(
                    func.avg(IntelligentDecision.confidence),
                    func.avg(IntelligentDecision.feedback_score)
                ).filter(
                    IntelligentDecision.tenant_id == tenant_id
                ).first()
                
                return {
                    'total': total,
                    'by_scenario': {s: c for s, c in scenario_stats},
                    'by_algorithm': {a: c for a, c in algo_stats},
                    'avg_confidence': float(avg_stats[0] or 0),
                    'avg_feedback_score': float(avg_stats[1] or 0)
                }
        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {'total': 0, 'by_scenario': {}, 'by_algorithm': {}, 'avg_confidence': 0, 'avg_feedback_score': 0}


# ==============================================================================
# 自适应优化仓库
# ==============================================================================

class AdaptiveOptimizationRepository:
    """自适应优化仓库"""
    
    def __init__(self, use_memory_storage: bool = False):
        self._use_memory_storage = use_memory_storage
        self._memory_storage: Dict[str, AdaptiveOptimizationData] = {}
        
        if not use_memory_storage:
            self._init_database()
    
    def _init_database(self):
        """初始化数据库连接"""
        try:
            from backend.core.database import get_db_session
            self._get_session = get_db_session
            logger.info("AdaptiveOptimizationRepository: Database initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize database: {e}")
            self._use_memory_storage = True
    
    def create(self, data: Dict[str, Any]) -> Any:
        """创建自适应优化记录"""
        if self._use_memory_storage:
            opt = AdaptiveOptimizationData(
                id=data.get('id', str(uuid.uuid4())),
                user_id=data.get('user_id', ''),
                tenant_id=data.get('tenant_id', ''),
                optimization_id=data.get('optimization_id', ''),
                parameter_name=data.get('parameter_name', ''),
                adjustment_strategy=data.get('adjustment_strategy', ''),
                adjustment_range=data.get('adjustment_range', {}),
                monitoring_metrics=data.get('monitoring_metrics', []),
                original_value=data.get('original_value'),
                optimized_value=data.get('optimized_value'),
                improvement_metric=data.get('improvement_metric', ''),
                improvement_value=data.get('improvement_value', 0.0),
                adjustment_reason=data.get('adjustment_reason', ''),
                tags=data.get('tags', [])
            )
            self._memory_storage[opt.id] = opt
            return opt
        
        try:
            from backend.schemas.training_models import AdaptiveOptimization
            from backend.core.database import get_db_session
            
            with get_db_session() as session:
                opt = AdaptiveOptimization(
                    user_id=data.get('user_id'),
                    tenant_id=data.get('tenant_id'),
                    optimization_id=data.get('optimization_id'),
                    parameter_name=data.get('parameter_name'),
                    adjustment_strategy=data.get('adjustment_strategy'),
                    adjustment_range=data.get('adjustment_range'),
                    monitoring_metrics=data.get('monitoring_metrics'),
                    original_value=data.get('original_value'),
                    optimized_value=data.get('optimized_value'),
                    improvement_metric=data.get('improvement_metric'),
                    improvement_value=data.get('improvement_value'),
                    adjustment_reason=data.get('adjustment_reason'),
                    tags=data.get('tags')
                )
                session.add(opt)
                session.commit()
                session.refresh(opt)
                return opt
        except Exception as e:
            logger.error(f"Failed to create adaptive optimization: {e}")
            raise
    
    def get_by_id(self, id: str) -> Optional[Any]:
        """根据ID获取记录"""
        if self._use_memory_storage:
            return self._memory_storage.get(id)
        
        try:
            from backend.schemas.training_models import AdaptiveOptimization
            from backend.core.database import get_db_session
            
            with get_db_session() as session:
                return session.query(AdaptiveOptimization).filter(
                    AdaptiveOptimization.id == id
                ).first()
        except Exception as e:
            logger.error(f"Failed to get adaptive optimization: {e}")
            return None
    
    def update(self, id: str, data: Dict[str, Any]) -> Optional[Any]:
        """更新记录"""
        if self._use_memory_storage:
            if id not in self._memory_storage:
                return None
            opt = self._memory_storage[id]
            for key, value in data.items():
                if hasattr(opt, key):
                    setattr(opt, key, value)
            opt.updated_at = datetime.now()
            return opt
        
        try:
            from backend.schemas.training_models import AdaptiveOptimization
            from backend.core.database import get_db_session
            
            with get_db_session() as session:
                opt = session.query(AdaptiveOptimization).filter(
                    AdaptiveOptimization.id == id
                ).first()
                
                if not opt:
                    return None
                
                for key, value in data.items():
                    if hasattr(opt, key):
                        setattr(opt, key, value)
                
                session.commit()
                session.refresh(opt)
                return opt
        except Exception as e:
            logger.error(f"Failed to update adaptive optimization: {e}")
            raise
    
    def list_by_tenant(self, tenant_id: str, user_id: Optional[str] = None,
                      parameter_name: Optional[str] = None,
                      limit: int = 20, offset: int = 0) -> Tuple[List[Any], int]:
        """按租户列出记录"""
        if self._use_memory_storage:
            results = []
            for opt in self._memory_storage.values():
                if opt.tenant_id != tenant_id:
                    continue
                if user_id and opt.user_id != user_id:
                    continue
                if parameter_name and opt.parameter_name != parameter_name:
                    continue
                results.append(opt)
            
            results.sort(key=lambda x: x.created_at or datetime.min, reverse=True)
            total = len(results)
            return results[offset:offset + limit], total
        
        try:
            from backend.schemas.training_models import AdaptiveOptimization
            from backend.core.database import get_db_session
            
            with get_db_session() as session:
                query = session.query(AdaptiveOptimization).filter(
                    AdaptiveOptimization.tenant_id == tenant_id
                )
                
                if user_id:
                    query = query.filter(AdaptiveOptimization.user_id == user_id)
                if parameter_name:
                    query = query.filter(AdaptiveOptimization.parameter_name == parameter_name)
                
                total = query.count()
                results = query.order_by(
                    AdaptiveOptimization.created_at.desc()
                ).offset(offset).limit(limit).all()
                
                return results, total
        except Exception as e:
            logger.error(f"Failed to list adaptive optimizations: {e}")
            return [], 0


# ==============================================================================
# 知识库仓库
# ==============================================================================

class KnowledgeBaseRepository:
    """知识库仓库"""
    
    def __init__(self, use_memory_storage: bool = False):
        self._use_memory_storage = use_memory_storage
        self._memory_storage: Dict[str, KnowledgeData] = {}
        
        if not use_memory_storage:
            self._init_database()
        else:
            self._init_default_knowledge()
    
    def _init_database(self):
        """初始化数据库连接"""
        try:
            from backend.core.database import get_db_session
            self._get_session = get_db_session
            logger.info("KnowledgeBaseRepository: Database initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize database: {e}")
            self._use_memory_storage = True
            self._init_default_knowledge()
    
    def _init_default_knowledge(self):
        """初始化默认知识"""
        default_knowledge = [
            KnowledgeData(
                id='kb_model_transformer',
                knowledge_type='model_architecture',
                category='nlp',
                title='Transformer架构知识',
                content={
                    'description': 'Transformer架构适用于自然语言处理任务',
                    'best_practices': ['使用预训练模型', '注意序列长度', '使用AdamW优化器'],
                    'parameters': {'attention_heads': 8, 'hidden_size': 768, 'layers': 12}
                },
                is_public=True
            ),
            KnowledgeData(
                id='kb_model_cnn',
                knowledge_type='model_architecture',
                category='cv',
                title='CNN架构知识',
                content={
                    'description': 'CNN架构适用于计算机视觉任务',
                    'best_practices': ['使用数据增强', '使用批归一化', '使用残差连接'],
                    'parameters': {'conv_layers': [64, 128, 256], 'kernel_size': 3}
                },
                is_public=True
            ),
            KnowledgeData(
                id='kb_hyperparameter_lr',
                knowledge_type='hyperparameter',
                category='training',
                title='学习率调优知识',
                content={
                    'description': '学习率是训练中最重要的超参数之一',
                    'strategies': ['warmup', 'cosine_annealing', 'reduce_on_plateau'],
                    'recommended_ranges': {'transformer': [1e-5, 5e-5], 'cnn': [1e-4, 1e-2]}
                },
                is_public=True
            )
        ]
        
        for k in default_knowledge:
            self._memory_storage[k.id] = k
    
    def create(self, data: Dict[str, Any]) -> Any:
        """创建知识记录"""
        if self._use_memory_storage:
            knowledge = KnowledgeData(
                id=data.get('id', str(uuid.uuid4())),
                user_id=data.get('user_id', ''),
                tenant_id=data.get('tenant_id', ''),
                knowledge_type=data.get('knowledge_type', ''),
                category=data.get('category', ''),
                title=data.get('title', ''),
                content=data.get('content', {}),
                related_entities=data.get('related_entities', []),
                relationships=data.get('relationships', []),
                source=data.get('source', ''),
                confidence=data.get('confidence', 1.0),
                is_public=data.get('is_public', False),
                tags=data.get('tags', [])
            )
            self._memory_storage[knowledge.id] = knowledge
            return knowledge
        
        try:
            from backend.schemas.training_models import KnowledgeBase
            from backend.core.database import get_db_session
            
            with get_db_session() as session:
                knowledge = KnowledgeBase(
                    user_id=data.get('user_id'),
                    tenant_id=data.get('tenant_id'),
                    knowledge_type=data.get('knowledge_type'),
                    category=data.get('category'),
                    title=data.get('title'),
                    content=data.get('content'),
                    related_entities=data.get('related_entities'),
                    relationships=data.get('relationships'),
                    source=data.get('source'),
                    confidence=data.get('confidence', 1.0),
                    is_public=data.get('is_public', False),
                    tags=data.get('tags')
                )
                session.add(knowledge)
                session.commit()
                session.refresh(knowledge)
                return knowledge
        except Exception as e:
            logger.error(f"Failed to create knowledge: {e}")
            raise
    
    def get_by_id(self, id: str) -> Optional[Any]:
        """根据ID获取知识"""
        if self._use_memory_storage:
            return self._memory_storage.get(id)
        
        try:
            from backend.schemas.training_models import KnowledgeBase
            from backend.core.database import get_db_session
            
            with get_db_session() as session:
                return session.query(KnowledgeBase).filter(
                    KnowledgeBase.id == id
                ).first()
        except Exception as e:
            logger.error(f"Failed to get knowledge: {e}")
            return None
    
    def update(self, id: str, data: Dict[str, Any]) -> Optional[Any]:
        """更新知识"""
        if self._use_memory_storage:
            if id not in self._memory_storage:
                return None
            knowledge = self._memory_storage[id]
            for key, value in data.items():
                if hasattr(knowledge, key):
                    setattr(knowledge, key, value)
            knowledge.updated_at = datetime.now()
            return knowledge
        
        try:
            from backend.schemas.training_models import KnowledgeBase
            from backend.core.database import get_db_session
            
            with get_db_session() as session:
                knowledge = session.query(KnowledgeBase).filter(
                    KnowledgeBase.id == id
                ).first()
                
                if not knowledge:
                    return None
                
                for key, value in data.items():
                    if hasattr(knowledge, key):
                        setattr(knowledge, key, value)
                
                session.commit()
                session.refresh(knowledge)
                return knowledge
        except Exception as e:
            logger.error(f"Failed to update knowledge: {e}")
            raise
    
    def search(self, tenant_id: str, query: str, 
              knowledge_type: Optional[str] = None,
              category: Optional[str] = None,
              limit: int = 20) -> List[Any]:
        """搜索知识"""
        if self._use_memory_storage:
            results = []
            query_lower = query.lower()
            
            for k in self._memory_storage.values():
                if not (k.is_public or k.tenant_id == tenant_id):
                    continue
                if knowledge_type and k.knowledge_type != knowledge_type:
                    continue
                if category and k.category != category:
                    continue
                
                # 简单的文本匹配
                if (query_lower in k.title.lower() or 
                    query_lower in str(k.content).lower()):
                    results.append(k)
            
            return results[:limit]
        
        try:
            from backend.schemas.training_models import KnowledgeBase
            from backend.core.database import get_db_session
            from sqlalchemy import or_
            
            with get_db_session() as session:
                q = session.query(KnowledgeBase).filter(
                    or_(
                        KnowledgeBase.is_public == True,
                        KnowledgeBase.tenant_id == tenant_id
                    ),
                    KnowledgeBase.is_active == True
                )
                
                if knowledge_type:
                    q = q.filter(KnowledgeBase.knowledge_type == knowledge_type)
                if category:
                    q = q.filter(KnowledgeBase.category == category)
                
                from sqlalchemy import String as SQLString
                
                # 简单的LIKE搜索
                q = q.filter(
                    or_(
                        KnowledgeBase.title.ilike(f'%{query}%'),
                        KnowledgeBase.content.cast(SQLString).ilike(f'%{query}%')
                    )
                )
                
                return q.limit(limit).all()
        except Exception as e:
            logger.error(f"Failed to search knowledge: {e}")
            return []
    
    def list_available(self, tenant_id: str, knowledge_type: Optional[str] = None,
                      category: Optional[str] = None,
                      limit: int = 20, offset: int = 0) -> Tuple[List[Any], int]:
        """列出可用知识"""
        if self._use_memory_storage:
            results = []
            for k in self._memory_storage.values():
                if not (k.is_public or k.tenant_id == tenant_id):
                    continue
                if knowledge_type and k.knowledge_type != knowledge_type:
                    continue
                if category and k.category != category:
                    continue
                results.append(k)
            
            total = len(results)
            return results[offset:offset + limit], total
        
        try:
            from backend.schemas.training_models import KnowledgeBase
            from backend.core.database import get_db_session
            from sqlalchemy import or_
            
            with get_db_session() as session:
                query = session.query(KnowledgeBase).filter(
                    or_(
                        KnowledgeBase.is_public == True,
                        KnowledgeBase.tenant_id == tenant_id
                    ),
                    KnowledgeBase.is_active == True
                )
                
                if knowledge_type:
                    query = query.filter(KnowledgeBase.knowledge_type == knowledge_type)
                if category:
                    query = query.filter(KnowledgeBase.category == category)
                
                total = query.count()
                results = query.offset(offset).limit(limit).all()
                
                return results, total
        except Exception as e:
            logger.error(f"Failed to list knowledge: {e}")
            return [], 0
    
    def increment_usage(self, id: str) -> bool:
        """增加使用计数"""
        if self._use_memory_storage:
            if id in self._memory_storage:
                self._memory_storage[id].usage_count += 1
                self._memory_storage[id].last_used_at = datetime.now()
                return True
            return False
        
        try:
            from backend.schemas.training_models import KnowledgeBase
            from backend.core.database import get_db_session
            
            with get_db_session() as session:
                k = session.query(KnowledgeBase).filter(
                    KnowledgeBase.id == id
                ).first()
                
                if k:
                    k.usage_count = (k.usage_count or 0) + 1
                    k.last_used_at = datetime.now()
                    session.commit()
                    return True
                return False
        except Exception as e:
            logger.error(f"Failed to increment usage: {e}")
            return False


# ==============================================================================
# 经验记录仓库
# ==============================================================================

class ExperienceRecordRepository:
    """经验记录仓库"""
    
    def __init__(self, use_memory_storage: bool = False):
        self._use_memory_storage = use_memory_storage
        self._memory_storage: Dict[str, ExperienceData] = {}
        
        if not use_memory_storage:
            self._init_database()
    
    def _init_database(self):
        """初始化数据库连接"""
        try:
            from backend.core.database import get_db_session
            self._get_session = get_db_session
            logger.info("ExperienceRecordRepository: Database initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize database: {e}")
            self._use_memory_storage = True
    
    def create(self, data: Dict[str, Any]) -> Any:
        """创建经验记录"""
        if self._use_memory_storage:
            experience = ExperienceData(
                id=data.get('id', str(uuid.uuid4())),
                user_id=data.get('user_id', ''),
                tenant_id=data.get('tenant_id', ''),
                experience_type=data.get('experience_type', ''),
                scenario=data.get('scenario', ''),
                context=data.get('context', {}),
                action=data.get('action', {}),
                result=data.get('result', {}),
                reward=data.get('reward', 0.0),
                decision_id=data.get('decision_id'),
                effectiveness=data.get('effectiveness'),
                lessons_learned=data.get('lessons_learned', ''),
                is_positive=data.get('is_positive', True),
                tags=data.get('tags', [])
            )
            self._memory_storage[experience.id] = experience
            return experience
        
        try:
            from backend.schemas.training_models import ExperienceRecord
            from backend.core.database import get_db_session
            
            with get_db_session() as session:
                experience = ExperienceRecord(
                    user_id=data.get('user_id'),
                    tenant_id=data.get('tenant_id'),
                    experience_type=data.get('experience_type'),
                    scenario=data.get('scenario'),
                    context=data.get('context'),
                    action=data.get('action'),
                    result=data.get('result'),
                    reward=data.get('reward'),
                    decision_id=data.get('decision_id'),
                    effectiveness=data.get('effectiveness'),
                    lessons_learned=data.get('lessons_learned'),
                    is_positive=data.get('is_positive', True),
                    tags=data.get('tags')
                )
                session.add(experience)
                session.commit()
                session.refresh(experience)
                return experience
        except Exception as e:
            logger.error(f"Failed to create experience: {e}")
            raise
    
    def get_by_id(self, id: str) -> Optional[Any]:
        """根据ID获取经验"""
        if self._use_memory_storage:
            return self._memory_storage.get(id)
        
        try:
            from backend.schemas.training_models import ExperienceRecord
            from backend.core.database import get_db_session
            
            with get_db_session() as session:
                return session.query(ExperienceRecord).filter(
                    ExperienceRecord.id == id
                ).first()
        except Exception as e:
            logger.error(f"Failed to get experience: {e}")
            return None
    
    def list_by_tenant(self, tenant_id: str, user_id: Optional[str] = None,
                      scenario: Optional[str] = None,
                      is_positive: Optional[bool] = None,
                      limit: int = 20, offset: int = 0) -> Tuple[List[Any], int]:
        """按租户列出经验"""
        if self._use_memory_storage:
            results = []
            for e in self._memory_storage.values():
                if e.tenant_id != tenant_id:
                    continue
                if user_id and e.user_id != user_id:
                    continue
                if scenario and e.scenario != scenario:
                    continue
                if is_positive is not None and e.is_positive != is_positive:
                    continue
                results.append(e)
            
            results.sort(key=lambda x: x.created_at or datetime.min, reverse=True)
            total = len(results)
            return results[offset:offset + limit], total
        
        try:
            from backend.schemas.training_models import ExperienceRecord
            from backend.core.database import get_db_session
            
            with get_db_session() as session:
                query = session.query(ExperienceRecord).filter(
                    ExperienceRecord.tenant_id == tenant_id
                )
                
                if user_id:
                    query = query.filter(ExperienceRecord.user_id == user_id)
                if scenario:
                    query = query.filter(ExperienceRecord.scenario == scenario)
                if is_positive is not None:
                    query = query.filter(ExperienceRecord.is_positive == is_positive)
                
                total = query.count()
                results = query.order_by(
                    ExperienceRecord.created_at.desc()
                ).offset(offset).limit(limit).all()
                
                return results, total
        except Exception as e:
            logger.error(f"Failed to list experiences: {e}")
            return [], 0
    
    def get_by_decision(self, decision_id: str) -> List[Any]:
        """获取决策关联的经验"""
        if self._use_memory_storage:
            return [e for e in self._memory_storage.values() if e.decision_id == decision_id]
        
        try:
            from backend.schemas.training_models import ExperienceRecord
            from backend.core.database import get_db_session
            
            with get_db_session() as session:
                return session.query(ExperienceRecord).filter(
                    ExperienceRecord.decision_id == decision_id
                ).all()
        except Exception as e:
            logger.error(f"Failed to get experiences by decision: {e}")
            return []


# ==============================================================================
# 仓库工厂函数
# ==============================================================================

_decision_repo_instance = None
_adaptive_opt_repo_instance = None
_knowledge_repo_instance = None
_experience_repo_instance = None


def get_intelligent_decision_repository(use_memory_storage: bool = False) -> IntelligentDecisionRepository:
    """获取智能决策仓库实例"""
    global _decision_repo_instance
    if _decision_repo_instance is None:
        _decision_repo_instance = IntelligentDecisionRepository(use_memory_storage)
    return _decision_repo_instance


def get_adaptive_optimization_repository(use_memory_storage: bool = False) -> AdaptiveOptimizationRepository:
    """获取自适应优化仓库实例"""
    global _adaptive_opt_repo_instance
    if _adaptive_opt_repo_instance is None:
        _adaptive_opt_repo_instance = AdaptiveOptimizationRepository(use_memory_storage)
    return _adaptive_opt_repo_instance


def get_knowledge_base_repository(use_memory_storage: bool = False) -> KnowledgeBaseRepository:
    """获取知识库仓库实例"""
    global _knowledge_repo_instance
    if _knowledge_repo_instance is None:
        _knowledge_repo_instance = KnowledgeBaseRepository(use_memory_storage)
    return _knowledge_repo_instance


def get_experience_record_repository(use_memory_storage: bool = False) -> ExperienceRecordRepository:
    """获取经验记录仓库实例"""
    global _experience_repo_instance
    if _experience_repo_instance is None:
        _experience_repo_instance = ExperienceRecordRepository(use_memory_storage)
    return _experience_repo_instance


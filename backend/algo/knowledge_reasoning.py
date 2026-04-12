"""
知识推理引擎 - 生产级实现

基于知识图谱和规则的智能推理系统，支持多种高级特性：

核心特性：
- 知识表示：实体、关系、规则、本体
- 前向链推理：从事实出发得出结论
- 后向链推理：从目标出发逆向搜索
- 概率推理：贝叶斯推理、不确定性传播
- 模糊推理：模糊规则、模糊匹配
- 语义相似度：基于嵌入的相似度计算
- 路径搜索：知识图谱路径发现
- 推理解释：可解释的推理过程
- 规则学习：从数据中学习新规则
- 缓存优化：推理结果缓存

使用示例：
    config = AlgorithmConfig(
        algorithm_type=AlgorithmType.KNOWLEDGE_REASONING,
        inference_depth=10,
        extra={
            "enable_fuzzy": True,
            "enable_probabilistic": True,
            "similarity_threshold": 0.7
        }
    )
    
    engine = KnowledgeReasoningEngine(config)
    result = engine.suggest(context)
"""

import logging
import re
import math
import hashlib
import json
from typing import Dict, List, Any, Optional, Tuple, Set, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
from datetime import datetime
from abc import ABC, abstractmethod
from functools import lru_cache
import heapq

import numpy as np

from .base import (
    BaseAlgorithm,
    AlgorithmType,
    AlgorithmConfig,
    AlgorithmContext,
    AlgorithmResult
)

logger = logging.getLogger(__name__)


# ==================== 枚举类型定义 ====================

class EntityType(Enum):
    """实体类型 - 扩展版"""
    # 模型相关
    MODEL = "model"
    MODEL_ARCHITECTURE = "model_architecture"
    LAYER = "layer"
    ACTIVATION = "activation"
    
    # 数据相关
    DATASET = "dataset"
    DATA_TYPE = "data_type"
    FEATURE = "feature"
    
    # 超参数相关
    HYPERPARAMETER = "hyperparameter"
    OPTIMIZER = "optimizer"
    SCHEDULER = "scheduler"
    
    # 任务相关
    TASK = "task"
    SUBTASK = "subtask"
    METRIC = "metric"
    
    # 资源相关
    RESOURCE = "resource"
    HARDWARE = "hardware"
    FRAMEWORK = "framework"
    
    # 策略相关
    STRATEGY = "strategy"
    TECHNIQUE = "technique"
    PATTERN = "pattern"
    
    # 概念相关
    CONCEPT = "concept"
    DOMAIN = "domain"


class RelationType(Enum):
    """关系类型 - 扩展版"""
    # 适用性关系
    SUITABLE_FOR = "suitable_for"
    RECOMMENDED_FOR = "recommended_for"
    OPTIMAL_FOR = "optimal_for"
    
    # 依赖关系
    REQUIRES = "requires"
    DEPENDS_ON = "depends_on"
    PREREQUISITE_OF = "prerequisite_of"
    
    # 效果关系
    IMPROVES = "improves"
    DEGRADES = "degrades"
    AFFECTS = "affects"
    
    # 兼容性关系
    COMPATIBLE_WITH = "compatible_with"
    CONFLICTS_WITH = "conflicts_with"
    REPLACES = "replaces"
    
    # 层级关系
    IS_A = "is_a"
    PART_OF = "part_of"
    HAS_PART = "has_part"
    INSTANCE_OF = "instance_of"
    
    # 比较关系
    BETTER_THAN = "better_than"
    WORSE_THAN = "worse_than"
    SIMILAR_TO = "similar_to"
    
    # 产出关系
    PRODUCES = "produces"
    CONSUMES = "consumes"
    TRANSFORMS = "transforms"


class ConditionOperator(Enum):
    """条件操作符"""
    EQUALS = "equals"
    NOT_EQUALS = "not_equals"
    GREATER_THAN = "gt"
    GREATER_THAN_OR_EQUAL = "gte"
    LESS_THAN = "lt"
    LESS_THAN_OR_EQUAL = "lte"
    CONTAINS = "contains"
    NOT_CONTAINS = "not_contains"
    IN = "in"
    NOT_IN = "not_in"
    MATCHES = "matches"  # 正则匹配
    RANGE = "range"
    EXISTS = "exists"
    NOT_EXISTS = "not_exists"
    FUZZY_EQUALS = "fuzzy_equals"  # 模糊匹配


class InferenceType(Enum):
    """推理类型"""
    FORWARD_CHAINING = "forward_chaining"
    BACKWARD_CHAINING = "backward_chaining"
    PROBABILISTIC = "probabilistic"
    FUZZY = "fuzzy"
    HYBRID = "hybrid"


# ==================== 数据结构定义 ====================

@dataclass
class Entity:
    """知识实体
    
    Attributes:
        entity_id: 实体唯一标识
        entity_type: 实体类型
        name: 实体名称
        properties: 属性字典
        confidence: 置信度 (0-1)
        embedding: 语义嵌入向量
        aliases: 别名列表
        description: 描述
        metadata: 元数据
    """
    entity_id: str
    entity_type: EntityType
    name: str
    properties: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    embedding: Optional[List[float]] = None
    aliases: List[str] = field(default_factory=list)
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def matches(self, query: Dict[str, Any], fuzzy: bool = False) -> Tuple[bool, float]:
        """检查是否匹配查询条件
        
        Args:
            query: 查询条件
            fuzzy: 是否启用模糊匹配
            
        Returns:
            (是否匹配, 匹配分数)
        """
        score = 1.0
        
        for key, value in query.items():
            if key == 'type':
                if self.entity_type.value != value:
                    if fuzzy:
                        score *= 0.5
                    else:
                        return False, 0.0
            elif key == 'name':
                if fuzzy:
                    name_score = self._fuzzy_match(value, self.name)
                    alias_scores = [self._fuzzy_match(value, a) for a in self.aliases]
                    max_score = max([name_score] + alias_scores)
                    if max_score < 0.5:
                        return False, 0.0
                    score *= max_score
                else:
                    if value.lower() not in self.name.lower():
                        if not any(value.lower() in a.lower() for a in self.aliases):
                            return False, 0.0
            elif key in self.properties:
                if fuzzy and isinstance(value, str) and isinstance(self.properties[key], str):
                    match_score = self._fuzzy_match(value, self.properties[key])
                    if match_score < 0.5:
                        return False, 0.0
                    score *= match_score
                elif self.properties[key] != value:
                    return False, 0.0
        
        return True, score * self.confidence
    
    def _fuzzy_match(self, s1: str, s2: str) -> float:
        """模糊字符串匹配（Levenshtein 相似度）"""
        s1, s2 = s1.lower(), s2.lower()
        if s1 == s2:
            return 1.0
        
        len1, len2 = len(s1), len(s2)
        if len1 == 0 or len2 == 0:
            return 0.0
        
        # 简化的编辑距离
        distance = self._levenshtein_distance(s1, s2)
        max_len = max(len1, len2)
        return 1.0 - (distance / max_len)
    
    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """计算 Levenshtein 距离"""
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'entity_id': self.entity_id,
            'entity_type': self.entity_type.value,
            'name': self.name,
            'properties': self.properties,
            'confidence': self.confidence,
            'aliases': self.aliases,
            'description': self.description
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Entity':
        """从字典创建"""
        return cls(
            entity_id=data['entity_id'],
            entity_type=EntityType(data['entity_type']),
            name=data['name'],
            properties=data.get('properties', {}),
            confidence=data.get('confidence', 1.0),
            aliases=data.get('aliases', []),
            description=data.get('description', '')
        )


@dataclass
class Relation:
    """知识关系 - 增强版"""
    source_id: str
    target_id: str
    relation_type: RelationType
    properties: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    weight: float = 1.0
    bidirectional: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __hash__(self):
        return hash((self.source_id, self.target_id, self.relation_type))
    
    def __eq__(self, other):
        if not isinstance(other, Relation):
            return False
        return (self.source_id == other.source_id and 
                self.target_id == other.target_id and
                self.relation_type == other.relation_type)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'source_id': self.source_id,
            'target_id': self.target_id,
            'relation_type': self.relation_type.value,
            'properties': self.properties,
            'confidence': self.confidence,
            'weight': self.weight,
            'bidirectional': self.bidirectional
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Relation':
        return cls(
            source_id=data['source_id'],
            target_id=data['target_id'],
            relation_type=RelationType(data['relation_type']),
            properties=data.get('properties', {}),
            confidence=data.get('confidence', 1.0),
            weight=data.get('weight', 1.0),
            bidirectional=data.get('bidirectional', False)
        )


@dataclass
class Condition:
    """规则条件 - 结构化"""
    field: str
    operator: ConditionOperator
    value: Any
    negated: bool = False
    weight: float = 1.0
    
    def evaluate(self, facts: Dict[str, Any]) -> Tuple[bool, float]:
        """评估条件
        
        Returns:
            (是否满足, 满足程度 0-1)
        """
        fact_value = facts.get(self.field)
        
        if self.operator == ConditionOperator.EXISTS:
            result = fact_value is not None
            return (not result if self.negated else result, 1.0 if result else 0.0)
        
        if self.operator == ConditionOperator.NOT_EXISTS:
            result = fact_value is None
            return (not result if self.negated else result, 1.0 if result else 0.0)
        
        if fact_value is None:
            return (self.negated, 0.0)
        
        result = False
        degree = 0.0
        
        if self.operator == ConditionOperator.EQUALS:
            result = fact_value == self.value
            degree = 1.0 if result else 0.0
        
        elif self.operator == ConditionOperator.NOT_EQUALS:
            result = fact_value != self.value
            degree = 1.0 if result else 0.0
        
        elif self.operator == ConditionOperator.GREATER_THAN:
            try:
                result = float(fact_value) > float(self.value)
                degree = 1.0 if result else 0.0
            except (ValueError, TypeError):
                return (self.negated, 0.0)
        
        elif self.operator == ConditionOperator.GREATER_THAN_OR_EQUAL:
            try:
                result = float(fact_value) >= float(self.value)
                degree = 1.0 if result else 0.0
            except (ValueError, TypeError):
                return (self.negated, 0.0)
        
        elif self.operator == ConditionOperator.LESS_THAN:
            try:
                result = float(fact_value) < float(self.value)
                degree = 1.0 if result else 0.0
            except (ValueError, TypeError):
                return (self.negated, 0.0)
        
        elif self.operator == ConditionOperator.LESS_THAN_OR_EQUAL:
            try:
                result = float(fact_value) <= float(self.value)
                degree = 1.0 if result else 0.0
            except (ValueError, TypeError):
                return (self.negated, 0.0)
        
        elif self.operator == ConditionOperator.CONTAINS:
            result = str(self.value) in str(fact_value)
            degree = 1.0 if result else 0.0
        
        elif self.operator == ConditionOperator.IN:
            if isinstance(self.value, (list, tuple, set)):
                result = fact_value in self.value
                degree = 1.0 if result else 0.0
            else:
                return (self.negated, 0.0)
        
        elif self.operator == ConditionOperator.MATCHES:
            try:
                result = bool(re.match(str(self.value), str(fact_value)))
                degree = 1.0 if result else 0.0
            except re.error:
                return (self.negated, 0.0)
        
        elif self.operator == ConditionOperator.RANGE:
            try:
                low, high = self.value
                val = float(fact_value)
                result = low <= val <= high
                if result:
                    # 计算在范围内的位置作为满足度
                    degree = 1.0 - abs(val - (low + high) / 2) / ((high - low) / 2)
                else:
                    degree = 0.0
            except (ValueError, TypeError):
                return (self.negated, 0.0)
        
        elif self.operator == ConditionOperator.FUZZY_EQUALS:
            if isinstance(fact_value, str) and isinstance(self.value, str):
                # 模糊字符串匹配
                similarity = self._string_similarity(fact_value, self.value)
                result = similarity >= 0.7
                degree = similarity
            else:
                result = fact_value == self.value
                degree = 1.0 if result else 0.0
        
        if self.negated:
            result = not result
            degree = 1.0 - degree
        
        return (result, degree * self.weight)
    
    def _string_similarity(self, s1: str, s2: str) -> float:
        """字符串相似度"""
        s1, s2 = s1.lower(), s2.lower()
        if s1 == s2:
            return 1.0
        
        len1, len2 = len(s1), len(s2)
        if len1 == 0 or len2 == 0:
            return 0.0
        
        # Jaccard 相似度（基于字符）
        set1, set2 = set(s1), set(s2)
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        return intersection / union if union > 0 else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'field': self.field,
            'operator': self.operator.value,
            'value': self.value,
            'negated': self.negated,
            'weight': self.weight
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Condition':
        return cls(
            field=data['field'],
            operator=ConditionOperator(data.get('operator', 'equals')),
            value=data['value'],
            negated=data.get('negated', False),
            weight=data.get('weight', 1.0)
        )


@dataclass
class Rule:
    """推理规则 - 增强版"""
    rule_id: str
    name: str
    conditions: List[Condition]
    conclusion: Dict[str, Any]
    confidence: float = 1.0
    priority: int = 0
    enabled: bool = True
    category: str = "general"
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    # 统计信息
    fire_count: int = 0
    success_count: int = 0
    last_fired: Optional[datetime] = None
    
    def evaluate(self, facts: Dict[str, Any]) -> Tuple[bool, float]:
        """评估规则条件
        
        Returns:
            (是否满足所有条件, 综合满足度)
        """
        if not self.enabled:
            return False, 0.0
        
        if not self.conditions:
            return True, self.confidence
        
        all_satisfied = True
        total_degree = 0.0
        total_weight = 0.0
        
        for condition in self.conditions:
            if isinstance(condition, dict):
                # 兼容旧格式
                condition = Condition.from_dict(condition)
            
            satisfied, degree = condition.evaluate(facts)
            
            if not satisfied:
                all_satisfied = False
            
            total_degree += degree * condition.weight
            total_weight += condition.weight
        
        avg_degree = total_degree / total_weight if total_weight > 0 else 0.0
        
        return all_satisfied, avg_degree * self.confidence
    
    def fire(self, success: bool = True) -> None:
        """记录规则触发"""
        self.fire_count += 1
        if success:
            self.success_count += 1
        self.last_fired = datetime.now()
    
    def get_success_rate(self) -> float:
        """获取成功率"""
        if self.fire_count == 0:
            return 0.0
        return self.success_count / self.fire_count
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'rule_id': self.rule_id,
            'name': self.name,
            'conditions': [c.to_dict() if isinstance(c, Condition) else c 
                         for c in self.conditions],
            'conclusion': self.conclusion,
            'confidence': self.confidence,
            'priority': self.priority,
            'enabled': self.enabled,
            'category': self.category,
            'description': self.description,
            'fire_count': self.fire_count,
            'success_count': self.success_count
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Rule':
        conditions = []
        for c in data.get('conditions', []):
            if isinstance(c, dict) and 'field' in c:
                conditions.append(Condition.from_dict(c))
            else:
                conditions.append(c)
        
        return cls(
            rule_id=data['rule_id'],
            name=data['name'],
            conditions=conditions,
            conclusion=data['conclusion'],
            confidence=data.get('confidence', 1.0),
            priority=data.get('priority', 0),
            enabled=data.get('enabled', True),
            category=data.get('category', 'general'),
            description=data.get('description', ''),
            fire_count=data.get('fire_count', 0),
            success_count=data.get('success_count', 0)
        )


@dataclass
class InferenceResult:
    """推理结果 - 增强版"""
    conclusion: Dict[str, Any]
    confidence: float
    reasoning_chain: List[str]
    supporting_facts: List[Dict[str, Any]]
    rules_fired: List[str] = field(default_factory=list)
    inference_type: InferenceType = InferenceType.FORWARD_CHAINING
    execution_time_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'conclusion': self.conclusion,
            'confidence': self.confidence,
            'reasoning_chain': self.reasoning_chain,
            'supporting_facts': self.supporting_facts,
            'rules_fired': self.rules_fired,
            'inference_type': self.inference_type.value,
            'execution_time_ms': self.execution_time_ms
        }


@dataclass
class PathSearchResult:
    """路径搜索结果"""
    path: List[str]  # 实体 ID 路径
    relations: List[str]  # 关系类型
    total_weight: float
    confidence: float


# ==================== 推理缓存 ====================

class InferenceCache:
    """推理结果缓存"""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        self._cache: Dict[str, Tuple[Any, datetime]] = {}
        self._max_size = max_size
        self._ttl_seconds = ttl_seconds
        self._hits = 0
        self._misses = 0
    
    def _make_key(self, facts: Dict[str, Any]) -> str:
        """生成缓存键"""
        sorted_items = sorted(facts.items(), key=lambda x: x[0])
        return hashlib.md5(json.dumps(sorted_items, default=str).encode()).hexdigest()
    
    def get(self, facts: Dict[str, Any]) -> Optional[Any]:
        """获取缓存结果"""
        key = self._make_key(facts)
        if key in self._cache:
            result, timestamp = self._cache[key]
            if (datetime.now() - timestamp).seconds < self._ttl_seconds:
                self._hits += 1
                return result
            else:
                del self._cache[key]
        self._misses += 1
        return None
    
    def set(self, facts: Dict[str, Any], result: Any) -> None:
        """设置缓存结果"""
        if len(self._cache) >= self._max_size:
            # 删除最旧的项
            oldest_key = min(self._cache, key=lambda k: self._cache[k][1])
            del self._cache[oldest_key]
        
        key = self._make_key(facts)
        self._cache[key] = (result, datetime.now())
    
    def clear(self) -> None:
        """清空缓存"""
        self._cache.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        total = self._hits + self._misses
        return {
            'size': len(self._cache),
            'max_size': self._max_size,
            'hits': self._hits,
            'misses': self._misses,
            'hit_rate': self._hits / total if total > 0 else 0.0
        }


class KnowledgeGraph:
    """知识图谱 - 生产级实现
    
    存储实体、关系和推理规则，支持多种查询和推理操作。
    
    特性：
    - 高效的索引结构
    - 路径搜索
    - 语义相似度查询
    - 知识导入/导出
    - 统计分析
    """
    
    def __init__(self):
        self._entities: Dict[str, Entity] = {}
        self._relations: List[Relation] = []
        self._rules: Dict[str, Rule] = {}
        
        # 多维索引
        self._entity_by_type: Dict[EntityType, List[str]] = defaultdict(list)
        self._entity_by_name: Dict[str, List[str]] = defaultdict(list)
        self._relations_by_source: Dict[str, List[Relation]] = defaultdict(list)
        self._relations_by_target: Dict[str, List[Relation]] = defaultdict(list)
        self._relations_by_type: Dict[RelationType, List[Relation]] = defaultdict(list)
        self._rules_by_category: Dict[str, List[str]] = defaultdict(list)
        
        # 邻接表（用于图算法）
        self._adjacency: Dict[str, Dict[str, List[Relation]]] = defaultdict(lambda: defaultdict(list))
        
        # 统计信息
        self._stats = {
            'entities_added': 0,
            'relations_added': 0,
            'rules_added': 0,
            'queries_executed': 0
        }
        
        # 初始化默认知识
        self._initialize_default_knowledge()
    
    def _initialize_default_knowledge(self):
        """初始化默认知识"""
        # 添加模型实体
        self._add_entity(Entity(
            entity_id="model_transformer",
            entity_type=EntityType.MODEL,
            name="Transformer",
            properties={
                "architecture": "transformer",
                "suitable_tasks": ["nlp", "text_classification", "translation"],
                "min_data_size": "medium",
                "compute_requirement": "high"
            }
        ))
        
        self._add_entity(Entity(
            entity_id="model_resnet",
            entity_type=EntityType.MODEL,
            name="ResNet",
            properties={
                "architecture": "cnn",
                "suitable_tasks": ["image_classification", "object_detection"],
                "min_data_size": "small",
                "compute_requirement": "medium"
            }
        ))
        
        self._add_entity(Entity(
            entity_id="model_mlp",
            entity_type=EntityType.MODEL,
            name="MLP",
            properties={
                "architecture": "mlp",
                "suitable_tasks": ["tabular", "regression", "classification"],
                "min_data_size": "small",
                "compute_requirement": "low"
            }
        ))
        
        self._add_entity(Entity(
            entity_id="model_lstm",
            entity_type=EntityType.MODEL,
            name="LSTM",
            properties={
                "architecture": "rnn",
                "suitable_tasks": ["sequence", "time_series", "nlp"],
                "min_data_size": "medium",
                "compute_requirement": "medium"
            }
        ))
        
        # 添加超参数实体
        self._add_entity(Entity(
            entity_id="hp_learning_rate",
            entity_type=EntityType.HYPERPARAMETER,
            name="学习率",
            properties={
                "parameter_name": "learning_rate",
                "default_range": [1e-5, 1e-1],
                "recommended_for_transformer": [1e-5, 5e-5],
                "recommended_for_cnn": [1e-4, 1e-2]
            }
        ))
        
        self._add_entity(Entity(
            entity_id="hp_batch_size",
            entity_type=EntityType.HYPERPARAMETER,
            name="批次大小",
            properties={
                "parameter_name": "batch_size",
                "default_range": [8, 256],
                "memory_dependent": True
            }
        ))
        
        # 添加任务实体
        self._add_entity(Entity(
            entity_id="task_nlp",
            entity_type=EntityType.TASK,
            name="自然语言处理",
            properties={
                "task_type": "nlp",
                "data_type": "text"
            }
        ))
        
        self._add_entity(Entity(
            entity_id="task_cv",
            entity_type=EntityType.TASK,
            name="计算机视觉",
            properties={
                "task_type": "cv",
                "data_type": "image"
            }
        ))
        
        # 添加关系
        self._add_relation(Relation(
            source_id="model_transformer",
            target_id="task_nlp",
            relation_type=RelationType.SUITABLE_FOR,
            confidence=0.95
        ))
        
        self._add_relation(Relation(
            source_id="model_resnet",
            target_id="task_cv",
            relation_type=RelationType.SUITABLE_FOR,
            confidence=0.92
        ))
        
        self._add_relation(Relation(
            source_id="model_transformer",
            target_id="hp_learning_rate",
            relation_type=RelationType.REQUIRES,
            properties={"recommended_value": 2e-5}
        ))
        
        # 添加推理规则
        self._add_rule(Rule(
            rule_id="rule_transformer_for_nlp",
            name="NLP任务推荐Transformer",
            conditions=[
                {"field": "data_type", "operator": "equals", "value": "text"},
                {"field": "data_size", "operator": "gte", "value": "medium"}
            ],
            conclusion={
                "recommendation": "transformer",
                "reason": "文本数据且数据量充足时，推荐使用Transformer架构"
            },
            confidence=0.9,
            priority=10
        ))
        
        self._add_rule(Rule(
            rule_id="rule_cnn_for_cv",
            name="CV任务推荐CNN",
            conditions=[
                {"field": "data_type", "operator": "equals", "value": "image"}
            ],
            conclusion={
                "recommendation": "cnn",
                "reason": "图像数据推荐使用CNN架构"
            },
            confidence=0.88,
            priority=9
        ))
        
        self._add_rule(Rule(
            rule_id="rule_small_data_mlp",
            name="小数据推荐MLP",
            conditions=[
                {"field": "data_size", "operator": "equals", "value": "small"},
                {"field": "data_type", "operator": "equals", "value": "tabular"}
            ],
            conclusion={
                "recommendation": "mlp",
                "reason": "小规模表格数据推荐使用简单的MLP"
            },
            confidence=0.82,
            priority=5
        ))
        
        self._add_rule(Rule(
            rule_id="rule_high_lr_for_cnn",
            name="CNN使用较高学习率",
            conditions=[
                {"field": "model_type", "operator": "equals", "value": "cnn"}
            ],
            conclusion={
                "learning_rate": 1e-3,
                "reason": "CNN模型通常使用较高的学习率"
            },
            confidence=0.85,
            priority=6
        ))
        
        self._add_rule(Rule(
            rule_id="rule_low_lr_for_transformer",
            name="Transformer使用较低学习率",
            conditions=[
                {"field": "model_type", "operator": "equals", "value": "transformer"}
            ],
            conclusion={
                "learning_rate": 2e-5,
                "reason": "Transformer模型推荐使用较低的学习率"
            },
            confidence=0.9,
            priority=8
        ))
        
        self._add_rule(Rule(
            rule_id="rule_large_batch_for_large_data",
            name="大数据使用大批次",
            conditions=[
                {"field": "data_size", "operator": "equals", "value": "large"},
                {"field": "gpu_memory", "operator": "gte", "value": 16}
            ],
            conclusion={
                "batch_size": 64,
                "reason": "大数据集且GPU显存充足时，使用较大批次加速训练"
            },
            confidence=0.8,
            priority=4
        ))
    
    def _add_entity(self, entity: Entity) -> None:
        """添加实体"""
        self._entities[entity.entity_id] = entity
        self._entity_by_type[entity.entity_type].append(entity.entity_id)
        self._entity_by_name[entity.name.lower()].append(entity.entity_id)
        
        # 添加别名索引
        for alias in entity.aliases:
            self._entity_by_name[alias.lower()].append(entity.entity_id)
        
        self._stats['entities_added'] += 1
    
    def _add_relation(self, relation: Relation) -> None:
        """添加关系"""
        self._relations.append(relation)
        self._relations_by_source[relation.source_id].append(relation)
        self._relations_by_target[relation.target_id].append(relation)
        self._relations_by_type[relation.relation_type].append(relation)
        
        # 更新邻接表
        self._adjacency[relation.source_id][relation.target_id].append(relation)
        if relation.bidirectional:
            self._adjacency[relation.target_id][relation.source_id].append(relation)
        
        self._stats['relations_added'] += 1
    
    def _add_rule(self, rule: Rule) -> None:
        """添加规则"""
        self._rules[rule.rule_id] = rule
        self._rules_by_category[rule.category].append(rule.rule_id)
        self._stats['rules_added'] += 1
    
    def add_entity(self, entity: Entity) -> None:
        """公开方法：添加实体"""
        self._add_entity(entity)
    
    def add_relation(self, relation: Relation) -> None:
        """公开方法：添加关系"""
        self._add_relation(relation)
    
    def add_rule(self, rule: Rule) -> None:
        """公开方法：添加规则"""
        self._add_rule(rule)
    
    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """获取实体"""
        self._stats['queries_executed'] += 1
        return self._entities.get(entity_id)
    
    def get_entities_by_type(self, entity_type: EntityType) -> List[Entity]:
        """获取某类型的所有实体"""
        entity_ids = self._entity_by_type.get(entity_type, [])
        return [self._entities[eid] for eid in entity_ids if eid in self._entities]
    
    def get_entities_by_name(self, name: str, fuzzy: bool = False) -> List[Entity]:
        """按名称获取实体"""
        self._stats['queries_executed'] += 1
        
        if fuzzy:
            results = []
            name_lower = name.lower()
            for entity in self._entities.values():
                matched, score = entity.matches({'name': name}, fuzzy=True)
                if matched and score > 0.5:
                    results.append((entity, score))
            results.sort(key=lambda x: x[1], reverse=True)
            return [e for e, _ in results]
        else:
            entity_ids = self._entity_by_name.get(name.lower(), [])
            return [self._entities[eid] for eid in entity_ids if eid in self._entities]
    
    def query_entities(
        self, 
        query: Dict[str, Any], 
        fuzzy: bool = False,
        limit: int = 100
    ) -> List[Tuple[Entity, float]]:
        """查询实体
        
        Args:
            query: 查询条件
            fuzzy: 是否启用模糊匹配
            limit: 返回数量限制
            
        Returns:
            (实体, 匹配分数) 列表
        """
        self._stats['queries_executed'] += 1
        
        results = []
        for entity in self._entities.values():
            matched, score = entity.matches(query, fuzzy=fuzzy)
            if matched:
                results.append((entity, score))
        
        # 按分数排序
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]
    
    def get_relations(
        self, 
        source_id: str = None, 
                     target_id: str = None,
        relation_type: RelationType = None,
        min_confidence: float = 0.0
    ) -> List[Relation]:
        """获取关系"""
        self._stats['queries_executed'] += 1
        
        if source_id:
            relations = self._relations_by_source.get(source_id, [])
        elif target_id:
            relations = self._relations_by_target.get(target_id, [])
        elif relation_type:
            relations = self._relations_by_type.get(relation_type, [])
        else:
            relations = self._relations
        
        # 过滤
        if relation_type and source_id:
            relations = [r for r in relations if r.relation_type == relation_type]
        
        if min_confidence > 0:
            relations = [r for r in relations if r.confidence >= min_confidence]
        
        return relations
    
    def get_neighbors(
        self, 
        entity_id: str, 
        relation_types: List[RelationType] = None,
        direction: str = "both"
    ) -> List[Tuple[Entity, Relation]]:
        """获取实体的邻居
        
        Args:
            entity_id: 实体 ID
            relation_types: 关系类型过滤
            direction: "outgoing", "incoming", or "both"
            
        Returns:
            (邻居实体, 关系) 列表
        """
        neighbors = []
        
        if direction in ["outgoing", "both"]:
            for rel in self._relations_by_source.get(entity_id, []):
                if relation_types and rel.relation_type not in relation_types:
                    continue
                target = self._entities.get(rel.target_id)
                if target:
                    neighbors.append((target, rel))
        
        if direction in ["incoming", "both"]:
            for rel in self._relations_by_target.get(entity_id, []):
                if relation_types and rel.relation_type not in relation_types:
                    continue
                source = self._entities.get(rel.source_id)
                if source:
                    neighbors.append((source, rel))
        
        return neighbors
    
    def find_path(
        self, 
        start_id: str, 
        end_id: str, 
        max_depth: int = 5,
        relation_types: List[RelationType] = None
    ) -> Optional[PathSearchResult]:
        """查找两个实体之间的路径（BFS）
        
        Args:
            start_id: 起始实体 ID
            end_id: 目标实体 ID
            max_depth: 最大搜索深度
            relation_types: 允许的关系类型
            
        Returns:
            路径搜索结果
        """
        if start_id == end_id:
            return PathSearchResult(
                path=[start_id],
                relations=[],
                total_weight=0.0,
                confidence=1.0
            )
        
        # BFS
        visited = {start_id}
        queue = [(start_id, [start_id], [], 0.0, 1.0)]  # (node, path, rels, weight, conf)
        
        while queue:
            current, path, rels, weight, conf = queue.pop(0)
            
            if len(path) > max_depth:
                continue
            
            for neighbor_id, relations in self._adjacency[current].items():
                if neighbor_id in visited:
                    continue
                
                for rel in relations:
                    if relation_types and rel.relation_type not in relation_types:
                        continue
                    
                    new_path = path + [neighbor_id]
                    new_rels = rels + [rel.relation_type.value]
                    new_weight = weight + rel.weight
                    new_conf = conf * rel.confidence
                    
                    if neighbor_id == end_id:
                        return PathSearchResult(
                            path=new_path,
                            relations=new_rels,
                            total_weight=new_weight,
                            confidence=new_conf
                        )
                    
                    visited.add(neighbor_id)
                    queue.append((neighbor_id, new_path, new_rels, new_weight, new_conf))
        
        return None
    
    def find_all_paths(
        self, 
        start_id: str, 
        end_id: str, 
        max_depth: int = 5,
        max_paths: int = 10
    ) -> List[PathSearchResult]:
        """查找所有路径（DFS）"""
        all_paths = []
        
        def dfs(current, path, rels, weight, conf, visited):
            if len(path) > max_depth or len(all_paths) >= max_paths:
                return
            
            if current == end_id:
                all_paths.append(PathSearchResult(
                    path=path.copy(),
                    relations=rels.copy(),
                    total_weight=weight,
                    confidence=conf
                ))
                return
            
            for neighbor_id, relations in self._adjacency[current].items():
                if neighbor_id in visited:
                    continue
                
                for rel in relations:
                    visited.add(neighbor_id)
                    path.append(neighbor_id)
                    rels.append(rel.relation_type.value)
                    
                    dfs(
                        neighbor_id, path, rels, 
                        weight + rel.weight, 
                        conf * rel.confidence, 
                        visited
                    )
                    
                    path.pop()
                    rels.pop()
                    visited.remove(neighbor_id)
        
        dfs(start_id, [start_id], [], 0.0, 1.0, {start_id})
        return all_paths
    
    def get_rules(
        self, 
        priority_threshold: int = 0,
        category: str = None,
        enabled_only: bool = True
    ) -> List[Rule]:
        """获取规则"""
        if category:
            rule_ids = self._rules_by_category.get(category, [])
            rules = [self._rules[rid] for rid in rule_ids if rid in self._rules]
        else:
            rules = list(self._rules.values())
        
        if enabled_only:
            rules = [r for r in rules if r.enabled]
        
        rules = [r for r in rules if r.priority >= priority_threshold]
        return sorted(rules, key=lambda x: x.priority, reverse=True)
    
    def compute_similarity(
        self, 
        entity1_id: str, 
        entity2_id: str
    ) -> float:
        """计算两个实体的相似度
        
        基于共同邻居和属性相似度
        """
        e1 = self._entities.get(entity1_id)
        e2 = self._entities.get(entity2_id)
        
        if not e1 or not e2:
            return 0.0
        
        # 类型相似度
        type_sim = 1.0 if e1.entity_type == e2.entity_type else 0.3
        
        # 属性相似度
        common_keys = set(e1.properties.keys()) & set(e2.properties.keys())
        if common_keys:
            prop_matches = sum(
                1 for k in common_keys if e1.properties[k] == e2.properties[k]
            )
            prop_sim = prop_matches / len(common_keys)
        else:
            prop_sim = 0.0
        
        # 邻居相似度（Jaccard）
        neighbors1 = set(self._adjacency[entity1_id].keys())
        neighbors2 = set(self._adjacency[entity2_id].keys())
        
        if neighbors1 or neighbors2:
            intersection = len(neighbors1 & neighbors2)
            union = len(neighbors1 | neighbors2)
            neighbor_sim = intersection / union if union > 0 else 0.0
        else:
            neighbor_sim = 0.0
        
        # 综合相似度
        return 0.3 * type_sim + 0.4 * prop_sim + 0.3 * neighbor_sim
    
    def get_stats(self) -> Dict[str, Any]:
        """获取知识图谱统计信息"""
        return {
            'total_entities': len(self._entities),
            'total_relations': len(self._relations),
            'total_rules': len(self._rules),
            'entity_types': {
                t.value: len(ids) for t, ids in self._entity_by_type.items()
            },
            'relation_types': {
                t.value: len(rels) for t, rels in self._relations_by_type.items()
            },
            'rule_categories': {
                c: len(ids) for c, ids in self._rules_by_category.items()
            },
            **self._stats
        }
    
    def export_to_dict(self) -> Dict[str, Any]:
        """导出知识图谱为字典"""
        return {
            'entities': [e.to_dict() for e in self._entities.values()],
            'relations': [r.to_dict() for r in self._relations],
            'rules': [r.to_dict() for r in self._rules.values()]
        }
    
    def import_from_dict(self, data: Dict[str, Any]) -> None:
        """从字典导入知识图谱"""
        for entity_data in data.get('entities', []):
            entity = Entity.from_dict(entity_data)
            self._add_entity(entity)
        
        for relation_data in data.get('relations', []):
            relation = Relation.from_dict(relation_data)
            self._add_relation(relation)
        
        for rule_data in data.get('rules', []):
            rule = Rule.from_dict(rule_data)
            self._add_rule(rule)


class KnowledgeReasoningEngine(BaseAlgorithm):
    """知识推理引擎 - 生产级实现
    
    结合知识图谱和规则进行智能推理。
    
    特性：
    - 前向链推理：从事实出发
    - 后向链推理：从目标出发
    - 概率推理：贝叶斯不确定性处理
    - 模糊推理：模糊规则匹配
    - 混合推理：多种推理策略组合
    - 推理解释：生成可解释的推理过程
    - 缓存优化：推理结果缓存
    """
    
    def __init__(self, config: Optional[AlgorithmConfig] = None):
        super().__init__(config)
        self._knowledge_graph = KnowledgeGraph()
        
        # 配置选项
        extra = config.extra if config and config.extra else {}
        self._enable_fuzzy = extra.get('enable_fuzzy', True)
        self._enable_probabilistic = extra.get('enable_probabilistic', True)
        self._similarity_threshold = extra.get('similarity_threshold', 0.7)
        self._use_cache = extra.get('use_cache', True)
        self._max_inference_depth = config.inference_depth if config else 10
        
        # 推理缓存
        self._cache = InferenceCache(
            max_size=extra.get('cache_size', 1000),
            ttl_seconds=extra.get('cache_ttl', 3600)
        )
        
        # 推理历史
        self._inference_history: List[InferenceResult] = []
        
    @property
    def algorithm_type(self) -> AlgorithmType:
        return AlgorithmType.KNOWLEDGE_REASONING
    
    @property
    def knowledge_graph(self) -> KnowledgeGraph:
        """获取知识图谱"""
        return self._knowledge_graph
    
    def initialize(self, context: AlgorithmContext) -> None:
        """初始化推理引擎
        
        Args:
            context: 算法上下文
        """
        super().initialize(context)
        
        # 从上下文中的知识更新图谱
        if 'knowledge' in context.metadata:
            self._update_knowledge_graph(context.metadata['knowledge'])
    
        self.logger.info(
            f"Knowledge reasoning engine initialized: "
            f"{len(self._knowledge_graph._entities)} entities, "
            f"{len(self._knowledge_graph._relations)} relations, "
            f"{len(self._knowledge_graph._rules)} rules"
        )
    
    def suggest(
        self, 
        context: AlgorithmContext,
        inference_type: InferenceType = InferenceType.HYBRID,
        goal: Optional[Dict[str, Any]] = None
    ) -> AlgorithmResult:
        """生成推理建议
        
        Args:
            context: 算法上下文
            inference_type: 推理类型
            goal: 可选的目标（用于后向链推理）
            
        Returns:
            算法结果
        """
        import time
        start_time = time.time()
        
        if not self._initialized:
            self.initialize(context)
        
        self._iteration_count += 1
        
        # 提取事实
        facts = self._extract_facts(context)
        
        # 检查缓存
        if self._use_cache:
            cached_result = self._cache.get(facts)
            if cached_result:
                self.logger.debug("Using cached inference result")
                return cached_result
        
        # 根据推理类型执行推理
        if inference_type == InferenceType.FORWARD_CHAINING:
            conclusions, reasoning_chain, confidence = self._forward_chaining(
                facts, fuzzy=self._enable_fuzzy
            )
            inference_result = InferenceResult(
                conclusion=self._synthesize_conclusions(conclusions, context)[0],
                confidence=confidence,
                reasoning_chain=reasoning_chain,
                supporting_facts=[{'key': k, 'value': v} for k, v in facts.items()],
                rules_fired=[c.get('rule_id', '') for c in conclusions],
                inference_type=inference_type
            )
        
        elif inference_type == InferenceType.BACKWARD_CHAINING:
            if not goal:
                # 如果没有指定目标，使用默认目标
                goal = {'recommendation': context.inputs.get('task_type', 'general')}
            
            proved, reasoning_chain, confidence = self._backward_chaining(goal, facts)
            inference_result = InferenceResult(
                conclusion=goal if proved else self._default_recommendation(context),
                confidence=confidence if proved else 0.5,
                reasoning_chain=reasoning_chain,
                supporting_facts=[{'key': k, 'value': v} for k, v in facts.items()],
                inference_type=inference_type
            )
        
        elif inference_type == InferenceType.PROBABILISTIC:
            conclusions, reasoning_chain, _ = self._forward_chaining(facts)
            
            # 对结论进行概率增强
            for conclusion in conclusions:
                if 'recommendation' in conclusion:
                    prob, _ = self._probabilistic_inference(facts, conclusion['recommendation'])
                    conclusion['probability'] = prob
            
            action, confidence = self._synthesize_conclusions(conclusions, context)
            inference_result = InferenceResult(
                conclusion=action,
                confidence=confidence,
                reasoning_chain=reasoning_chain,
                supporting_facts=[{'key': k, 'value': v} for k, v in facts.items()],
                inference_type=inference_type
            )
        
        else:  # HYBRID
            inference_result = self._hybrid_inference(facts, goal)
        
        # 记录执行时间
        execution_time = (time.time() - start_time) * 1000
        inference_result.execution_time_ms = execution_time
        
        # 记录历史
        self._inference_history.append(inference_result)
        
        # 缓存结果
        if self._use_cache:
            self._cache.set(facts, inference_result)
        
        # 合并结论生成推荐
        action = inference_result.conclusion
        confidence = inference_result.confidence
        
        # 生成推理说明
        reasoning_steps = [
            f"推理类型: {inference_type.value}",
            f"提取的事实: {len(facts)} 条",
            *[f"  {k}={v}" for k, v in list(facts.items())[:5]],
            f"触发的规则: {len(inference_result.rules_fired)} 条",
            f"推理置信度: {confidence:.4f}",
            f"执行时间: {execution_time:.2f}ms",
            "---推理链---",
            *inference_result.reasoning_chain[:10]
        ]
        
        # 生成备选方案
        if inference_type in [InferenceType.FORWARD_CHAINING, InferenceType.HYBRID]:
            conclusions, _, _ = self._forward_chaining(facts)
            alternatives = self._generate_alternatives(conclusions, exclude=action)
        else:
            alternatives = []
        
        result = self._build_result(
            action=action,
            confidence=confidence,
            reasoning=f"基于{inference_type.value}推理：分析了{len(facts)}条事实，"
                     f"触发{len(inference_result.rules_fired)}条规则",
            alternatives=alternatives,
            reasoning_steps=reasoning_steps,
            debug_info={
                'facts_count': len(facts),
                'rules_fired': len(inference_result.rules_fired),
                'knowledge_entities': len(self._knowledge_graph._entities),
                'knowledge_relations': len(self._knowledge_graph._relations),
                'inference_type': inference_type.value,
                'execution_time_ms': execution_time,
                'cache_stats': self._cache.get_stats() if self._use_cache else None
            }
        )
        
        return result
    
    def update(self, action: Dict[str, Any], reward: float,
               context: Optional[AlgorithmContext] = None) -> None:
        """更新知识图谱
        
        根据反馈调整规则置信度。
        
        Args:
            action: 执行的动作
            reward: 获得的奖励
            context: 可选的上下文
        """
        self._record_observation(action, reward)
        
        # 根据奖励调整规则置信度
        # 如果奖励高，增加相关规则的置信度
        for rule in self._knowledge_graph._rules.values():
            # 检查规则是否与动作相关
            if self._rule_matches_action(rule, action):
                adjustment = 0.01 * (reward - 0.5)  # 奖励>0.5时增加，否则减少
                rule.confidence = max(0.1, min(1.0, rule.confidence + adjustment))
    
    def query_knowledge(self, query: str) -> Dict[str, Any]:
        """查询知识图谱
        
        Args:
            query: 查询字符串
            
        Returns:
            查询结果
        """
        # 解析查询
        query_lower = query.lower()
        
        entities = []
        relationships = []
        recommendations = []
        
        # 关键词匹配
        keywords = {
            'model': EntityType.MODEL,
            'hyperparameter': EntityType.HYPERPARAMETER,
            'task': EntityType.TASK,
            'dataset': EntityType.DATASET
        }
        
        for keyword, entity_type in keywords.items():
            if keyword in query_lower:
                matched_entities = self._knowledge_graph.get_entities_by_type(entity_type)
                for entity in matched_entities:
                    entities.append({
                        'id': entity.entity_id,
                        'type': entity.entity_type.value,
                        'name': entity.name,
                        'properties': entity.properties
                    })
                    
                    # 获取相关关系
                    relations = self._knowledge_graph.get_relations(source_id=entity.entity_id)
                    for rel in relations:
                        target = self._knowledge_graph.get_entity(rel.target_id)
                        relationships.append({
                            'source': entity.name,
                            'relation': rel.relation_type.value,
                            'target': target.name if target else rel.target_id,
                            'confidence': rel.confidence
                        })
        
        # 提取推荐
        for rule in self._knowledge_graph.get_rules():
            if any(kw in query_lower for kw in rule.name.lower().split()):
                if 'recommendation' in rule.conclusion:
                    recommendations.append(rule.conclusion.get('reason', rule.name))
        
        return {
            'query': query,
            'entities': entities[:10],
            'relationships': relationships[:10],
            'recommendations': recommendations[:5]
        }
    
    def _extract_facts(self, context: AlgorithmContext) -> Dict[str, Any]:
        """从上下文提取事实
        
        Args:
            context: 算法上下文
            
        Returns:
            事实字典
        """
        facts = {}
        
        # 从inputs提取
        for key, value in context.inputs.items():
            facts[key] = value
        
        # 从constraints提取
        for key, value in context.constraints.items():
            facts[f"constraint_{key}"] = value
        
        # 从metadata提取
        if context.metadata:
            for key, value in context.metadata.items():
                if key != 'knowledge':
                    facts[f"meta_{key}"] = value
        
        return facts
    
    def _forward_chaining(
        self, 
        facts: Dict[str, Any],
        fuzzy: bool = False
    ) -> Tuple[List[Dict[str, Any]], List[str], float]:
        """前向链推理
        
        从事实出发，应用规则得出结论。
        
        Args:
            facts: 初始事实
            fuzzy: 是否启用模糊匹配
            
        Returns:
            (结论列表, 推理链, 综合置信度)
        """
        import time
        start_time = time.time()
        
        conclusions = []
        reasoning_chain = []
        working_facts = dict(facts)
        fired_rules = set()
        total_confidence = 0.0
        
        # 获取所有规则
        rules = self._knowledge_graph.get_rules(enabled_only=True)
        
        # 迭代推理直到没有新结论
        for iteration in range(self._max_inference_depth):
            new_conclusions = []
            
            for rule in rules:
                if rule.rule_id in fired_rules:
                    continue
                
                # 评估规则条件
                if isinstance(rule, Rule):
                    satisfied, degree = rule.evaluate(working_facts)
                else:
                    satisfied = self._evaluate_conditions(rule.conditions, working_facts)
                    degree = rule.confidence if satisfied else 0.0
                
                # 模糊推理：即使不完全满足也可能触发
                if fuzzy and not satisfied and degree > 0.5:
                    satisfied = True
                    degree *= 0.8  # 降低置信度
                
                if satisfied:
                    # 规则触发
                    rule.fire(success=True)
                    fired_rules.add(rule.rule_id)
                    
                    conclusion = {
                        **rule.conclusion,
                        'rule_id': rule.rule_id,
                        'rule_name': rule.name,
                        'confidence': degree,
                        'iteration': iteration
                    }
                    
                    new_conclusions.append(conclusion)
                    total_confidence += degree
                    
                    reasoning_chain.append(
                        f"[迭代{iteration}] 规则 '{rule.name}' 触发 (置信度={degree:.2f}): "
                        f"{rule.conclusion.get('reason', str(rule.conclusion))}"
                    )
                        
                    # 将结论加入工作事实
                    for key, value in rule.conclusion.items():
                        if key not in ['reason', 'recommendation']:
                            working_facts[key] = value
            
            if not new_conclusions:
                break
            
            conclusions.extend(new_conclusions)
        
        execution_time = (time.time() - start_time) * 1000
        avg_confidence = total_confidence / len(conclusions) if conclusions else 0.0
        
        self.logger.debug(
            f"Forward chaining completed: {len(conclusions)} conclusions, "
            f"{len(fired_rules)} rules fired, {execution_time:.2f}ms"
        )
        
        return conclusions, reasoning_chain, avg_confidence
    
    def _backward_chaining(
        self, 
        goal: Dict[str, Any],
        facts: Dict[str, Any],
        max_depth: int = 10
    ) -> Tuple[bool, List[str], float]:
        """后向链推理
        
        从目标出发，逆向搜索支持的事实和规则。
        
        Args:
            goal: 目标结论
            facts: 当前已知事实
            max_depth: 最大搜索深度
            
        Returns:
            (是否能达成目标, 推理链, 置信度)
        """
        reasoning_chain = []
        visited_goals = set()
        
        def prove(current_goal: Dict[str, Any], depth: int) -> Tuple[bool, float]:
            if depth > max_depth:
                return False, 0.0
            
            # 创建目标的字符串表示用于去重
            goal_key = json.dumps(current_goal, sort_keys=True, default=str)
            if goal_key in visited_goals:
                return False, 0.0
            visited_goals.add(goal_key)
            
            # 检查目标是否已在事实中
            for key, value in current_goal.items():
                if key in facts:
                    if facts[key] == value:
                        reasoning_chain.append(f"事实验证: {key}={value}")
                        return True, 1.0
            
            # 查找能产生目标的规则
            for rule in self._knowledge_graph.get_rules():
                # 检查规则结论是否包含目标
                conclusion_matches = all(
                    rule.conclusion.get(k) == v 
                    for k, v in current_goal.items() 
                    if k not in ['reason', 'recommendation']
                )
                
                if conclusion_matches:
                    reasoning_chain.append(f"尝试规则: {rule.name}")
                    
                    # 尝试证明规则的所有条件
                    all_conditions_met = True
                    total_confidence = rule.confidence
                    
                    for condition in rule.conditions:
                        if isinstance(condition, dict):
                            field = condition.get('field')
                            value = condition.get('value')
                        else:
                            field = condition.field
                            value = condition.value
                        
                        # 检查事实
                        if field in facts:
                            if isinstance(condition, dict):
                                cond = Condition.from_dict(condition)
                            else:
                                cond = condition
                            satisfied, degree = cond.evaluate(facts)
                            if satisfied:
                                total_confidence *= degree
                                continue
                        
                        # 递归证明子目标
                        sub_goal = {field: value}
                        sub_proved, sub_conf = prove(sub_goal, depth + 1)
                        
                        if sub_proved:
                            total_confidence *= sub_conf
                        else:
                            all_conditions_met = False
                            break
                    
                    if all_conditions_met:
                        reasoning_chain.append(f"规则 '{rule.name}' 满足所有条件")
                        return True, total_confidence
            
            return False, 0.0
        
        proved, confidence = prove(goal, 0)
        return proved, reasoning_chain, confidence
    
    def _probabilistic_inference(
        self, 
        facts: Dict[str, Any],
        query: str
    ) -> Tuple[float, List[str]]:
        """概率推理（简化的贝叶斯推理）
        
        Args:
            facts: 当前事实
            query: 查询目标
            
        Returns:
            (概率, 推理链)
        """
        reasoning_chain = []
        
        # 收集相关证据
        evidence_probs = []
        
        # 从相关规则收集概率
        for rule in self._knowledge_graph.get_rules():
            if query in str(rule.conclusion):
                # 评估规则条件
                if isinstance(rule, Rule):
                    satisfied, degree = rule.evaluate(facts)
                else:
                    satisfied = self._evaluate_conditions(rule.conditions, facts)
                    degree = rule.confidence if satisfied else 0.0
                
                if satisfied or degree > 0.3:
                    # P(query|evidence) ∝ P(evidence|query) * P(query)
                    prior = rule.confidence
                    likelihood = degree
                    posterior = prior * likelihood
                    evidence_probs.append(posterior)
                    
                    reasoning_chain.append(
                        f"规则 '{rule.name}': P={posterior:.3f} "
                        f"(prior={prior:.3f}, likelihood={likelihood:.3f})"
                    )
        
        # 合并概率（假设独立）
        if evidence_probs:
            # Noisy-OR 组合
            combined_prob = 1.0 - np.prod([1 - p for p in evidence_probs])
        else:
            combined_prob = 0.0
        
        reasoning_chain.append(f"综合概率: P({query})={combined_prob:.3f}")
        
        return combined_prob, reasoning_chain
    
    def _hybrid_inference(
        self, 
        facts: Dict[str, Any],
        goal: Optional[Dict[str, Any]] = None
    ) -> InferenceResult:
        """混合推理
        
        结合前向链、后向链和概率推理
        """
        import time
        start_time = time.time()
        
        all_conclusions = []
        all_reasoning = []
        total_confidence = 0.0
        rules_fired = []
        
        # 1. 前向链推理
        forward_conclusions, forward_chain, forward_conf = self._forward_chaining(
            facts, fuzzy=self._enable_fuzzy
        )
        all_conclusions.extend(forward_conclusions)
        all_reasoning.extend(forward_chain)
        rules_fired.extend([c.get('rule_id', '') for c in forward_conclusions])
        
        if forward_conclusions:
            total_confidence = forward_conf
        
        # 2. 如果有特定目标，进行后向链推理
        if goal:
            proved, backward_chain, backward_conf = self._backward_chaining(goal, facts)
            all_reasoning.extend(backward_chain)
            
            if proved:
                all_conclusions.append({
                    **goal,
                    'proved_by': 'backward_chaining',
                    'confidence': backward_conf
                })
                total_confidence = max(total_confidence, backward_conf)
        
        # 3. 概率推理增强
        if self._enable_probabilistic and all_conclusions:
            for conclusion in all_conclusions:
                if 'recommendation' in conclusion:
                    prob, prob_chain = self._probabilistic_inference(
                        facts, conclusion['recommendation']
                    )
                    conclusion['probability'] = prob
                    all_reasoning.extend(prob_chain)
        
        execution_time = (time.time() - start_time) * 1000
        
        # 合成最终结果
        final_conclusion = self._synthesize_conclusions(all_conclusions, AlgorithmContext(
            inputs=facts, search_space={}, constraints={}, observations=[]
        ))[0]
        
        return InferenceResult(
            conclusion=final_conclusion,
            confidence=total_confidence,
            reasoning_chain=all_reasoning,
            supporting_facts=[{'key': k, 'value': v} for k, v in facts.items()],
            rules_fired=list(set(rules_fired)),
            inference_type=InferenceType.HYBRID,
            execution_time_ms=execution_time
        )
    
    def _evaluate_conditions(
        self, 
        conditions: List[Any], 
        facts: Dict[str, Any]
    ) -> bool:
        """评估规则条件（兼容旧格式）"""
        for condition in conditions:
            if isinstance(condition, Condition):
                satisfied, _ = condition.evaluate(facts)
                if not satisfied:
                    return False
            elif isinstance(condition, dict):
                field = condition.get('field')
                operator = condition.get('operator', 'equals')
                value = condition.get('value')
                
                fact_value = facts.get(field)
                if fact_value is None:
                    return False
                
                if operator == 'equals':
                    if fact_value != value:
                        return False
                elif operator == 'not_equals':
                    if fact_value == value:
                        return False
                elif operator == 'contains':
                    if value not in str(fact_value):
                        return False
                elif operator in ['gte', 'gt']:
                    try:
                        if float(fact_value) < float(value):
                            return False
                    except (ValueError, TypeError):
                        return False
                elif operator in ['lte', 'lt']:
                    try:
                        if float(fact_value) > float(value):
                            return False
                    except (ValueError, TypeError):
                        return False
                elif operator == 'in':
                    if fact_value not in value:
                        return False
        
        return True
    
    def _synthesize_conclusions(self, conclusions: List[Dict[str, Any]], 
                               context: AlgorithmContext) -> Tuple[Dict[str, Any], float]:
        """合成结论生成推荐
        
        Args:
            conclusions: 结论列表
            context: 算法上下文
            
        Returns:
            推荐动作和置信度
        """
        if not conclusions:
            # 没有结论时使用默认推荐
            return self._default_recommendation(context), 0.5
        
        # 按置信度加权合并结论
        action = {}
        total_confidence = 0
        weighted_confidence = 0
        
        for conclusion in conclusions:
            conf = conclusion.get('confidence', 0.5)
            total_confidence += conf
            weighted_confidence += conf ** 2
            
            for key, value in conclusion.items():
                if key in ['rule_id', 'rule_name', 'confidence', 'reason']:
                    continue
                
                if key not in action:
                    action[key] = value
                elif key == 'recommendation':
                    # 保留置信度最高的推荐
                    if conf > conclusion.get('confidence', 0):
                        action[key] = value
        
        # 计算综合置信度
        if conclusions:
            avg_confidence = total_confidence / len(conclusions)
            confidence = min(0.95, avg_confidence * (1 + len(conclusions) * 0.05))
        else:
            confidence = 0.5
        
        return action, confidence
    
    def _default_recommendation(self, context: AlgorithmContext) -> Dict[str, Any]:
        """默认推荐
        
        当规则推理无结果时使用。
        
        Args:
            context: 算法上下文
            
        Returns:
            默认推荐
        """
        data_type = context.inputs.get('data_type', 'tabular')
        data_size = context.inputs.get('data_size', 'medium')
        
        if data_type == 'text':
            return {
                'recommendation': 'transformer',
                'learning_rate': 2e-5,
                'batch_size': 32,
                'reason': '默认推荐：文本数据使用Transformer'
            }
        elif data_type == 'image':
            return {
                'recommendation': 'cnn',
                'learning_rate': 1e-3,
                'batch_size': 32,
                'reason': '默认推荐：图像数据使用CNN'
            }
        else:
            return {
                'recommendation': 'mlp',
                'learning_rate': 1e-3,
                'batch_size': 64,
                'reason': '默认推荐：通用数据使用MLP'
            }
    
    def _rule_matches_action(self, rule: Rule, action: Dict[str, Any]) -> bool:
        """检查规则是否与动作相关
        
        Args:
            rule: 规则
            action: 动作
            
        Returns:
            是否相关
        """
        for key, value in rule.conclusion.items():
            if key in action:
                if action[key] == value:
                    return True
        return False
    
    def _update_knowledge_graph(self, knowledge: Dict[str, Any]) -> None:
        """更新知识图谱
        
        Args:
            knowledge: 新知识
        """
        # 添加实体
        if 'entities' in knowledge:
            for entity_data in knowledge['entities']:
                try:
                    entity = Entity(
                        entity_id=entity_data.get('id', str(hash(str(entity_data)))),
                        entity_type=EntityType(entity_data.get('type', 'model')),
                        name=entity_data.get('name', ''),
                        properties=entity_data.get('properties', {}),
                        confidence=entity_data.get('confidence', 1.0)
                    )
                    self._knowledge_graph._add_entity(entity)
                except (ValueError, KeyError) as e:
                    logger.warning(f"Failed to add entity: {e}")
        
        # 添加关系
        if 'relations' in knowledge:
            for relation_data in knowledge['relations']:
                try:
                    relation = Relation(
                        source_id=relation_data['source'],
                        target_id=relation_data['target'],
                        relation_type=RelationType(relation_data.get('type', 'suitable_for')),
                        properties=relation_data.get('properties', {}),
                        confidence=relation_data.get('confidence', 1.0)
                    )
                    self._knowledge_graph._add_relation(relation)
                except (ValueError, KeyError) as e:
                    logger.warning(f"Failed to add relation: {e}")
        
        # 添加规则
        if 'rules' in knowledge:
            for rule_data in knowledge['rules']:
                try:
                    rule = Rule(
                        rule_id=rule_data.get('id', str(hash(str(rule_data)))),
                        name=rule_data.get('name', ''),
                        conditions=rule_data.get('conditions', []),
                        conclusion=rule_data.get('conclusion', {}),
                        confidence=rule_data.get('confidence', 1.0),
                        priority=rule_data.get('priority', 0)
                    )
                    self._knowledge_graph._add_rule(rule)
                except (ValueError, KeyError) as e:
                    logger.warning(f"Failed to add rule: {e}")
    
    def _generate_alternatives(self, conclusions: List[Dict[str, Any]],
                              exclude: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """生成备选方案
        
        Args:
            conclusions: 结论列表
            exclude: 要排除的动作
            
        Returns:
            备选方案列表
        """
        alternatives = []
        
        for conclusion in conclusions:
            action = {k: v for k, v in conclusion.items() 
                     if k not in ['rule_id', 'rule_name', 'confidence']}
            
            if exclude and action == exclude:
                continue
            
            alternatives.append({
                'action': action,
                'confidence': conclusion.get('confidence', 0.5),
                'rule_name': conclusion.get('rule_name', ''),
                'reasoning': conclusion.get('reason', '')
            })
        
        # 按置信度排序
        alternatives.sort(key=lambda x: x['confidence'], reverse=True)
        
        return alternatives[:2]

    # ==================== 规则学习 ====================
    
    def learn_rules_from_observations(
        self, 
        min_support: float = 0.1,
        min_confidence: float = 0.7
    ) -> List[Rule]:
        """从历史观测中学习新规则
        
        使用简化的关联规则挖掘
        
        Args:
            min_support: 最小支持度
            min_confidence: 最小置信度
            
        Returns:
            学习到的规则列表
        """
        if len(self._observations) < 5:
            self.logger.warning("Not enough observations for rule learning")
            return []
        
        learned_rules = []
        
        # 统计特征-结果关联
        feature_value_counts = defaultdict(lambda: defaultdict(int))
        outcome_counts = defaultdict(int)
        total = len(self._observations)
        
        for obs in self._observations:
            action = obs.get('action', {})
            reward = obs.get('reward', 0)
            outcome = 'success' if reward > 0.5 else 'failure'
            outcome_counts[outcome] += 1
            
            for key, value in action.items():
                feature_value_counts[(key, str(value))][outcome] += 1
        
        # 生成规则
        for (feature, value), outcomes in feature_value_counts.items():
            total_count = sum(outcomes.values())
            support = total_count / total
            
            if support < min_support:
                continue
            
            for outcome, count in outcomes.items():
                confidence = count / total_count
                
                if confidence >= min_confidence:
                    rule = Rule(
                        rule_id=f"learned_rule_{feature}_{value}_{outcome}",
                        name=f"自动学习规则: {feature}={value} -> {outcome}",
                        conditions=[
                            Condition(
                                field=feature,
                                operator=ConditionOperator.EQUALS,
                                value=value
                            )
                        ],
                        conclusion={
                            'predicted_outcome': outcome,
                            'reason': f"基于{count}次观测学习，支持度={support:.2f}，置信度={confidence:.2f}"
                        },
                        confidence=confidence,
                        priority=int(confidence * 10),
                        category='learned'
                    )
                    learned_rules.append(rule)
                    self._knowledge_graph._add_rule(rule)
        
        self.logger.info(f"Learned {len(learned_rules)} rules from {total} observations")
        return learned_rules
    
    def refine_rules(self, feedback: Dict[str, Any]) -> None:
        """根据反馈优化规则
        
        Args:
            feedback: 反馈信息，包含 rule_id 和 success
        """
        rule_id = feedback.get('rule_id')
        success = feedback.get('success', False)
        
        if rule_id and rule_id in self._knowledge_graph._rules:
            rule = self._knowledge_graph._rules[rule_id]
            rule.fire(success=success)
            
            # 根据成功率调整置信度
            success_rate = rule.get_success_rate()
            if rule.fire_count >= 10:
                # 置信度向成功率靠拢
                rule.confidence = 0.7 * rule.confidence + 0.3 * success_rate
                
                # 如果成功率太低，禁用规则
                if success_rate < 0.3 and rule.fire_count >= 20:
                    rule.enabled = False
                    self.logger.info(f"Rule '{rule.name}' disabled due to low success rate")
    
    # ==================== 知识查询增强 ====================
    
    def semantic_search(
        self, 
        query: str, 
        entity_types: List[EntityType] = None,
        top_k: int = 10
    ) -> List[Tuple[Entity, float]]:
        """语义搜索实体
        
        Args:
            query: 查询字符串
            entity_types: 限制实体类型
            top_k: 返回数量
            
        Returns:
            (实体, 相似度分数) 列表
        """
        results = []
        
        for entity in self._knowledge_graph._entities.values():
            if entity_types and entity.entity_type not in entity_types:
                continue
            
            # 计算查询与实体的相似度
            score = 0.0
            
            # 名称匹配
            if query.lower() in entity.name.lower():
                score += 0.5
            
            # 别名匹配
            for alias in entity.aliases:
                if query.lower() in alias.lower():
                    score += 0.3
                    break
            
            # 属性匹配
            for key, value in entity.properties.items():
                if query.lower() in str(value).lower():
                    score += 0.2
                    break
            
            # 描述匹配
            if entity.description and query.lower() in entity.description.lower():
                score += 0.1
            
            if score > 0:
                results.append((entity, score * entity.confidence))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    def explain_inference(
        self, 
        inference_result: InferenceResult
    ) -> Dict[str, Any]:
        """生成推理解释
        
        Args:
            inference_result: 推理结果
            
        Returns:
            解释字典
        """
        explanation = {
            'summary': '',
            'confidence_explanation': '',
            'supporting_evidence': [],
            'rules_explanation': [],
            'alternative_conclusions': [],
            'uncertainty_factors': []
        }
        
        # 摘要
        conclusion = inference_result.conclusion
        explanation['summary'] = (
            f"推理得出结论: {conclusion.get('recommendation', conclusion)}, "
            f"置信度: {inference_result.confidence:.2%}"
        )
        
        # 置信度解释
        if inference_result.confidence >= 0.9:
            explanation['confidence_explanation'] = "高置信度：多条规则支持，证据充分"
        elif inference_result.confidence >= 0.7:
            explanation['confidence_explanation'] = "中高置信度：主要证据支持结论"
        elif inference_result.confidence >= 0.5:
            explanation['confidence_explanation'] = "中等置信度：部分证据支持，存在不确定性"
        else:
            explanation['confidence_explanation'] = "低置信度：证据不足，建议谨慎采纳"
        
        # 支持证据
        for fact in inference_result.supporting_facts:
            explanation['supporting_evidence'].append(
                f"{fact['key']}: {fact['value']}"
            )
        
        # 规则解释
        for rule_id in inference_result.rules_fired:
            rule = self._knowledge_graph._rules.get(rule_id)
            if rule:
                explanation['rules_explanation'].append({
                    'rule_name': rule.name,
                    'description': rule.description or rule.conclusion.get('reason', ''),
                    'confidence': rule.confidence,
                    'success_rate': rule.get_success_rate()
                })
        
        # 不确定性因素
        if inference_result.confidence < 0.7:
            explanation['uncertainty_factors'].append("置信度较低")
        if len(inference_result.rules_fired) < 2:
            explanation['uncertainty_factors'].append("触发规则较少")
        
        return explanation
    
    # ==================== 状态管理 ====================
    
    def get_state(self) -> Dict[str, Any]:
        """获取当前状态"""
        return {
            'observations': self._observations.copy(),
            'iteration_count': self._iteration_count,
            'knowledge_graph': self._knowledge_graph.export_to_dict(),
            'inference_history_count': len(self._inference_history),
            'cache_stats': self._cache.get_stats()
        }
    
    def load_state(self, state: Dict[str, Any]) -> None:
        """从状态恢复"""
        self._observations = state.get('observations', [])
        self._iteration_count = state.get('iteration_count', 0)
        
        if 'knowledge_graph' in state:
            self._knowledge_graph.import_from_dict(state['knowledge_graph'])
        
        self.logger.info(f"State loaded with {len(self._observations)} observations")
    
    def clear_cache(self) -> None:
        """清空缓存"""
        self._cache.clear()
    
    # ==================== 可视化支持 ====================
    
    def get_visualization_data(self) -> Dict[str, Any]:
        """获取可视化数据"""
        data = {
            'entities': [],
            'relations': [],
            'rules': [],
            'inference_history': [],
            'statistics': {}
        }
        
        # 实体数据
        for entity in self._knowledge_graph._entities.values():
            data['entities'].append({
                'id': entity.entity_id,
                'type': entity.entity_type.value,
                'name': entity.name,
                'confidence': entity.confidence
            })
        
        # 关系数据
        for relation in self._knowledge_graph._relations:
            source = self._knowledge_graph.get_entity(relation.source_id)
            target = self._knowledge_graph.get_entity(relation.target_id)
            data['relations'].append({
                'source': relation.source_id,
                'source_name': source.name if source else relation.source_id,
                'target': relation.target_id,
                'target_name': target.name if target else relation.target_id,
                'type': relation.relation_type.value,
                'confidence': relation.confidence
            })
        
        # 规则数据
        for rule in self._knowledge_graph._rules.values():
            data['rules'].append({
                'id': rule.rule_id,
                'name': rule.name,
                'category': rule.category,
                'confidence': rule.confidence,
                'priority': rule.priority,
                'enabled': rule.enabled,
                'fire_count': rule.fire_count,
                'success_rate': rule.get_success_rate()
            })
        
        # 推理历史
        for result in self._inference_history[-20:]:  # 最近 20 条
            data['inference_history'].append({
                'conclusion': result.conclusion,
                'confidence': result.confidence,
                'rules_fired': len(result.rules_fired),
                'inference_type': result.inference_type.value,
                'execution_time_ms': result.execution_time_ms
            })
        
        # 统计信息
        data['statistics'] = self._knowledge_graph.get_stats()
        data['statistics']['total_inferences'] = len(self._inference_history)
        data['statistics']['cache'] = self._cache.get_stats()
        
        return data
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """获取优化摘要"""
        if not self._inference_history:
            return {'status': 'no_inferences'}
        
        confidences = [r.confidence for r in self._inference_history]
        execution_times = [r.execution_time_ms for r in self._inference_history]
        
        return {
            'status': 'active',
            'total_inferences': len(self._inference_history),
            'total_observations': len(self._observations),
            'knowledge_stats': self._knowledge_graph.get_stats(),
            'inference_stats': {
                'avg_confidence': sum(confidences) / len(confidences),
                'max_confidence': max(confidences),
                'min_confidence': min(confidences),
                'avg_execution_time_ms': sum(execution_times) / len(execution_times)
            },
            'cache_stats': self._cache.get_stats(),
            'top_rules': self._get_top_rules(5)
        }
    
    def _get_top_rules(self, n: int = 5) -> List[Dict[str, Any]]:
        """获取最常触发的规则"""
        rules = list(self._knowledge_graph._rules.values())
        rules.sort(key=lambda r: r.fire_count, reverse=True)
        
        return [
            {
                'name': r.name,
                'fire_count': r.fire_count,
                'success_rate': r.get_success_rate(),
                'confidence': r.confidence
            }
            for r in rules[:n]
        ]

"""
智能化决策服务
提供AI驱动的自动化和知识图谱驱动功能，支持租户隔离和数据库持久化。
通过 algo 模块提供生产级的决策算法能力。
通过 langgraph 模块提供高级逻辑推理和多步骤 Agent 能力。
"""
import json
import logging
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Dict, List, Any, Optional, Callable

# 算法模块
from backend.algo import (
    AlgorithmFactory,
    AlgorithmConfig,
    AlgorithmContext as AlgoContext,
    AlgorithmType
)
# LangGraph Agent 模块
from backend.algo.langgraph import (
    # 状态
    AgentState,
    AgentStatus,
    # 工具
    Tool,
    ToolRegistry,
    tool,
    ToolCategory,
    # Agent
    AgentConfig,
    AgentRole,
    ReflexionAgent,
    MultiAgentSystem,
    # 工厂
    create_react_agent,
    create_plan_execute_agent,
    # 检查点
    MemoryCheckpointer,
    # 内置工具
    get_builtin_tools
)

logger = logging.getLogger(__name__)


class DecisionScenario(Enum):
    """决策场景"""
    DATA_PREPROCESSING = "data_preprocessing"     # 数据预处理策略选择
    MODEL_ARCHITECTURE = "model_architecture"     # 模型架构推荐
    HYPERPARAMETER_INIT = "hyperparameter_init"   # 超参数初始化
    RESOURCE_ALLOCATION = "resource_allocation"   # 资源分配决策
    TRAINING_STRATEGY = "training_strategy"       # 训练策略选择
    MODEL_SELECTION = "model_selection"           # 模型选择
    FEATURE_ENGINEERING = "feature_engineering"   # 特征工程


class DecisionAlgorithm(Enum):
    """决策算法"""
    REINFORCEMENT_LEARNING = "reinforcement_learning"  # 强化学习
    MULTI_ARMED_BANDIT = "multi_armed_bandit"         # 多臂老虎机
    BAYESIAN_OPTIMIZATION = "bayesian_optimization"   # 贝叶斯优化
    GENETIC_ALGORITHM = "genetic_algorithm"           # 遗传算法
    RULE_BASED = "rule_based"                         # 基于规则
    KNOWLEDGE_DRIVEN = "knowledge_driven"             # 知识驱动
    REACT_AGENT = "react_agent"                       # ReAct Agent
    PLAN_EXECUTE_AGENT = "plan_execute_agent"         # 计划执行 Agent
    REFLEXION_AGENT = "reflexion_agent"               # 反思 Agent
    MULTI_AGENT = "multi_agent"                       # 多 Agent 系统


class AgentMode(Enum):
    """Agent 执行模式"""
    SINGLE_STEP = "single_step"           # 单步推理
    MULTI_STEP = "multi_step"             # 多步推理
    PLAN_EXECUTE = "plan_execute"         # 计划执行
    REFLEXION = "reflexion"               # 反思模式
    COLLABORATIVE = "collaborative"        # 协作模式


@dataclass
class DecisionContext:
    """决策上下文"""
    scenario: DecisionScenario
    inputs: Dict[str, Any]
    constraints: Dict[str, Any] = field(default_factory=dict)
    history: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class DecisionResult:
    """决策结果"""
    decision_id: str
    scenario: str
    recommended_action: Any
    confidence: float
    reasoning: str
    alternatives: List[Dict[str, Any]]
    execution_plan: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class AdaptiveConfiguration:
    """自适应配置"""
    parameter_name: str
    current_value: Any
    adjustment_strategy: str
    adjustment_range: Optional[Dict[str, Any]] = None
    monitoring_metrics: List[str] = field(default_factory=list)


@dataclass
class AdaptiveOptimizationResult:
    """自适应优化结果"""
    optimization_id: str
    parameter_name: str
    original_value: Any
    optimized_value: Any
    improvement_metric: str
    improvement_value: float
    adjustment_reason: str
    timestamp: datetime


@dataclass
class AgentExecutionResult:
    """Agent 执行结果"""
    execution_id: str
    agent_type: str
    input_query: str
    final_answer: Optional[str]
    reasoning_steps: List[Dict[str, Any]]
    tool_calls: List[Dict[str, Any]]
    iterations: int
    status: str
    confidence: float
    execution_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentTask:
    """Agent 任务定义"""
    task_id: str
    task_type: str
    description: str
    inputs: Dict[str, Any]
    constraints: Dict[str, Any] = field(default_factory=dict)
    required_tools: List[str] = field(default_factory=list)
    max_iterations: int = 10
    timeout: float = 300.0


class IntelligentDecisionService:
    """智能化决策服务
    
    支持两种模式：
    1. 内存模式 (use_memory_storage=True): 数据存储在内存中，适用于测试
    2. 数据库模式 (use_memory_storage=False): 数据持久化到数据库
    
    所有操作都基于租户维度进行隔离。
    """

    def __init__(self, use_memory_storage: bool = True):
        """初始化智能化决策服务"""
        self.logger = logging.getLogger(__name__)
        self._use_memory_storage = use_memory_storage
        self._init_repositories()
        self._init_algorithms()
        
        # 内存中的决策历史（兼容旧代码）
        self.decision_history = []
        self.knowledge_base = {}

    def _init_repositories(self):
        """初始化数据访问层"""
        try:
            from backend.repositories.intelligent_decision_repository import (
                get_intelligent_decision_repository,
                get_adaptive_optimization_repository,
                get_knowledge_base_repository,
                get_experience_record_repository
            )
            
            self.decision_repo = get_intelligent_decision_repository(self._use_memory_storage)
            self.adaptive_opt_repo = get_adaptive_optimization_repository(self._use_memory_storage)
            self.knowledge_repo = get_knowledge_base_repository(self._use_memory_storage)
            self.experience_repo = get_experience_record_repository(self._use_memory_storage)
            
            self.logger.info(f"IntelligentDecisionService repositories initialized (memory={self._use_memory_storage})")
            
        except ImportError as e:
            self.logger.error(f"Failed to import repositories: {e}")
            raise
    
    def _init_algorithms(self):
        """初始化决策算法实例"""
        # 算法实例缓存
        self._algorithms: Dict[str, Any] = {}
        
        # Agent 实例缓存
        self._agents: Dict[str, Any] = {}
        
        # Agent 检查点器
        self._agent_checkpointer = MemoryCheckpointer()
        
        # 工具注册表
        self._tool_registry = ToolRegistry()
        self._init_decision_tools()
        
        # 场景到算法类型的映射
        self._scenario_algorithm_map = {
            DecisionScenario.DATA_PREPROCESSING: AlgorithmType.BAYESIAN_OPTIMIZATION,
            DecisionScenario.MODEL_ARCHITECTURE: AlgorithmType.KNOWLEDGE_REASONING,
            DecisionScenario.HYPERPARAMETER_INIT: AlgorithmType.MULTI_ARMED_BANDIT,
            DecisionScenario.RESOURCE_ALLOCATION: AlgorithmType.GENETIC_ALGORITHM,
            DecisionScenario.TRAINING_STRATEGY: AlgorithmType.REINFORCEMENT_LEARNING,
            DecisionScenario.MODEL_SELECTION: AlgorithmType.MULTI_ARMED_BANDIT,
            DecisionScenario.FEATURE_ENGINEERING: AlgorithmType.KNOWLEDGE_REASONING,
        }
        
        # 场景到 Agent 模式的映射（复杂场景使用 Agent）
        self._scenario_agent_map = {
            DecisionScenario.MODEL_ARCHITECTURE: AgentMode.MULTI_STEP,
            DecisionScenario.TRAINING_STRATEGY: AgentMode.PLAN_EXECUTE,
            DecisionScenario.FEATURE_ENGINEERING: AgentMode.REFLEXION,
        }
        
        self.logger.info("Algorithm and Agent instances will be created on-demand")
    
    def _init_decision_tools(self):
        """初始化决策相关的工具"""
        # 注册内置工具
        for builtin_tool in get_builtin_tools():
            self._tool_registry.register(builtin_tool)
        
        # 注册决策专用工具
        self._register_decision_tools()
    
    def get_available_algorithms(self) -> List[str]:
        """获取可用算法列表
        
        Returns:
            List[str]: 算法名称列表
        """
        return [algo.value for algo in DecisionAlgorithm]

    def _register_decision_tools(self):
        """注册决策专用工具"""
        
        @tool(
            name="analyze_data_characteristics",
            description="分析数据集的特征和统计信息",
            category=ToolCategory.DATA,
            register=False
        )
        def analyze_data_characteristics(data_info: str) -> str:
            """分析数据特征"""
            try:
                info = json.loads(data_info) if isinstance(data_info, str) else data_info
                
                analysis = {
                    "data_type": info.get("data_type", "unknown"),
                    "size": info.get("size", "medium"),
                    "features": info.get("features", []),
                    "recommendations": []
                }
                
                data_type = analysis["data_type"]
                if data_type == "text":
                    analysis["recommendations"].extend([
                        "建议使用 Transformer 架构",
                        "推荐 BERT 或 GPT 系列预训练模型",
                        "考虑使用分词器进行预处理"
                    ])
                elif data_type == "image":
                    analysis["recommendations"].extend([
                        "建议使用 CNN 架构",
                        "推荐 ResNet 或 EfficientNet",
                        "考虑数据增强策略"
                    ])
                elif data_type == "tabular":
                    analysis["recommendations"].extend([
                        "建议使用 XGBoost 或 LightGBM",
                        "考虑特征工程",
                        "推荐使用交叉验证"
                    ])
                
                return json.dumps(analysis, ensure_ascii=False, indent=2)
            except Exception as e:
                return f"分析失败: {str(e)}"
        
        @tool(
            name="recommend_hyperparameters",
            description="根据模型类型和数据规模推荐超参数",
            category=ToolCategory.COMPUTATION,
            register=False
        )
        def recommend_hyperparameters(model_type: str, data_size: str = "medium") -> str:
            """推荐超参数"""
            params = {
                "transformer": {
                    "small": {"learning_rate": 5e-5, "batch_size": 32, "epochs": 5, "warmup_ratio": 0.1},
                    "medium": {"learning_rate": 2e-5, "batch_size": 16, "epochs": 3, "warmup_ratio": 0.06},
                    "large": {"learning_rate": 1e-5, "batch_size": 8, "epochs": 2, "warmup_ratio": 0.1}
                },
                "cnn": {
                    "small": {"learning_rate": 1e-3, "batch_size": 64, "epochs": 30},
                    "medium": {"learning_rate": 1e-3, "batch_size": 32, "epochs": 50},
                    "large": {"learning_rate": 5e-4, "batch_size": 16, "epochs": 100}
                },
                "mlp": {
                    "small": {"learning_rate": 1e-3, "batch_size": 128, "epochs": 50},
                    "medium": {"learning_rate": 1e-3, "batch_size": 64, "epochs": 100},
                    "large": {"learning_rate": 5e-4, "batch_size": 32, "epochs": 200}
                }
            }
            
            model_params = params.get(model_type.lower(), params["mlp"])
            size_params = model_params.get(data_size, model_params["medium"])
            
            return json.dumps({
                "model_type": model_type,
                "data_size": data_size,
                "recommended_params": size_params,
                "optimizer": "adamw" if model_type.lower() == "transformer" else "adam"
            }, ensure_ascii=False, indent=2)
        
        @tool(
            name="estimate_resources",
            description="估算训练所需的计算资源",
            category=ToolCategory.COMPUTATION,
            register=False
        )
        def estimate_resources(model_complexity: str, data_size: str) -> str:
            """估算资源需求"""
            resource_map = {
                ("high", "large"): {"cpu": 16, "memory": 64, "gpu": 4, "gpu_memory": 24},
                ("high", "medium"): {"cpu": 8, "memory": 32, "gpu": 2, "gpu_memory": 16},
                ("high", "small"): {"cpu": 4, "memory": 16, "gpu": 1, "gpu_memory": 12},
                ("medium", "large"): {"cpu": 8, "memory": 32, "gpu": 2, "gpu_memory": 16},
                ("medium", "medium"): {"cpu": 4, "memory": 16, "gpu": 1, "gpu_memory": 12},
                ("medium", "small"): {"cpu": 2, "memory": 8, "gpu": 1, "gpu_memory": 8},
                ("low", "large"): {"cpu": 4, "memory": 16, "gpu": 1, "gpu_memory": 12},
                ("low", "medium"): {"cpu": 2, "memory": 8, "gpu": 1, "gpu_memory": 8},
                ("low", "small"): {"cpu": 2, "memory": 4, "gpu": 0, "gpu_memory": 0}
            }
            
            key = (model_complexity.lower(), data_size.lower())
            resources = resource_map.get(key, resource_map[("medium", "medium")])
            
            return json.dumps({
                "model_complexity": model_complexity,
                "data_size": data_size,
                "estimated_resources": resources,
                "estimated_time_hours": resources["gpu"] * 2 + 1
            }, ensure_ascii=False, indent=2)
        
        @tool(
            name="query_knowledge_base",
            description="查询知识库获取相关信息",
            category=ToolCategory.RETRIEVAL,
            register=False
        )
        def query_knowledge_base(query: str, category: str = None) -> str:
            """查询知识库"""
            # 模拟知识库查询
            knowledge_items = []
            
            query_lower = query.lower()
            
            if "transformer" in query_lower or "text" in query_lower:
                knowledge_items.append({
                    "title": "Transformer 最佳实践",
                    "content": "使用预训练模型、学习率预热、梯度累积"
                })
            
            if "cnn" in query_lower or "image" in query_lower:
                knowledge_items.append({
                    "title": "CNN 训练技巧",
                    "content": "数据增强、批量归一化、残差连接"
                })
            
            if "hyperparameter" in query_lower or "超参数" in query_lower:
                knowledge_items.append({
                    "title": "超参数调优策略",
                    "content": "贝叶斯优化、网格搜索、早停策略"
                })
            
            if not knowledge_items:
                knowledge_items.append({
                    "title": "通用建议",
                    "content": "从简单模型开始，逐步增加复杂度"
                })
            
            return json.dumps(knowledge_items, ensure_ascii=False, indent=2)
        
        @tool(
            name="evaluate_decision",
            description="评估决策方案的可行性和风险",
            category=ToolCategory.COMPUTATION,
            register=False
        )
        def evaluate_decision(decision: str) -> str:
            """评估决策"""
            try:
                decision_data = json.loads(decision) if isinstance(decision, str) else decision
                
                evaluation = {
                    "feasibility": 0.85,
                    "risk_level": "low",
                    "confidence": 0.80,
                    "potential_issues": [],
                    "recommendations": []
                }
                
                # 基于决策内容进行评估
                if "learning_rate" in decision_data:
                    lr = decision_data["learning_rate"]
                    if lr > 0.01:
                        evaluation["potential_issues"].append("学习率较高，可能导致训练不稳定")
                        evaluation["risk_level"] = "medium"
                    elif lr < 1e-6:
                        evaluation["potential_issues"].append("学习率过低，可能导致收敛缓慢")
                
                if "batch_size" in decision_data:
                    bs = decision_data["batch_size"]
                    if bs < 8:
                        evaluation["recommendations"].append("考虑使用更大的批量大小以提高训练稳定性")
                
                return json.dumps(evaluation, ensure_ascii=False, indent=2)
            except Exception as e:
                return f"评估失败: {str(e)}"
        
        # 注册工具
        self._tool_registry.register(analyze_data_characteristics._tool)
        self._tool_registry.register(recommend_hyperparameters._tool)
        self._tool_registry.register(estimate_resources._tool)
        self._tool_registry.register(query_knowledge_base._tool)
        self._tool_registry.register(evaluate_decision._tool)

    # ==========================================================================
    # 智能决策
    # ==========================================================================

    def make_intelligent_decision(self, context: DecisionContext, 
                                 tenant_id: str = None, 
                                 user_id: str = None) -> DecisionResult:
        """
        智能决策 - 使用生产级算法模块
        
        Args:
            context: 决策上下文
            tenant_id: 租户ID
            user_id: 用户ID
            
        Returns:
            DecisionResult: 决策结果
        """
        try:
            self.logger.info(f"开始智能决策，场景: {context.scenario.value}")
            
            # 根据场景选择决策算法类型
            algorithm_type = self._select_decision_algorithm(context.scenario)
            
            # 获取或创建算法实例
            algorithm = self._get_or_create_algorithm(context.scenario, algorithm_type)
            
            # 构建算法上下文
            algo_context = self._build_algorithm_context(context, tenant_id)
            
            # 执行算法获取建议
            algo_result = algorithm.suggest(algo_context)
            
            # 生成决策ID
            decision_id = f"decision_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{str(uuid.uuid4())[:8]}"
            
            # 生成执行计划
            execution_plan = self._generate_execution_plan(context, algo_result.recommended_action)
            
            # 创建决策结果
            result = DecisionResult(
                decision_id=decision_id,
                scenario=context.scenario.value,
                recommended_action=algo_result.recommended_action,
                confidence=algo_result.confidence,
                reasoning=algo_result.reasoning,
                alternatives=algo_result.alternatives,
                execution_plan=execution_plan,
                metadata={
                    "algorithm": algorithm_type.value,
                    "timestamp": datetime.utcnow().isoformat(),
                    "iterations": algo_result.iterations,
                    "convergence": algo_result.convergence,
                    "reasoning_steps": algo_result.reasoning_steps,
                    "debug_info": algo_result.debug_info
                }
            )
            
            # 持久化决策记录
            if tenant_id and user_id:
                self._save_decision(tenant_id, user_id, context, result, algorithm_type)
            
            # 记录到内存历史（兼容旧代码）
            self.decision_history.append({
                "decision_id": decision_id,
                "scenario": context.scenario.value,
                "inputs": context.inputs,
                "result": algo_result.recommended_action,
                "confidence": algo_result.confidence,
                "timestamp": datetime.utcnow().isoformat()
            })
            
            self.logger.info(f"智能决策完成，决策ID: {decision_id}, 算法: {algorithm_type.value}")
            return result
            
        except Exception as e:
            self.logger.error(f"智能决策失败: {str(e)}")
            # 回退到规则决策
            return self._fallback_decision(context, tenant_id, user_id, str(e))
    
    def _get_or_create_algorithm(self, scenario: DecisionScenario, 
                                algorithm_type: AlgorithmType) -> Any:
        """获取或创建算法实例
        
        Args:
            scenario: 决策场景
            algorithm_type: 算法类型
            
        Returns:
            算法实例
        """
        cache_key = f"{scenario.value}:{algorithm_type.value}"
        
        if cache_key not in self._algorithms:
            # 创建场景特定的配置
            config = self._create_algorithm_config(scenario, algorithm_type)
            self._algorithms[cache_key] = AlgorithmFactory.create(
                algorithm_type, config, use_cache=False
            )
            self.logger.info(f"Created algorithm instance: {cache_key}")
        
        return self._algorithms[cache_key]
    
    def _create_algorithm_config(self, scenario: DecisionScenario,
                                algorithm_type: AlgorithmType) -> AlgorithmConfig:
        """创建场景优化的算法配置
        
        Args:
            scenario: 决策场景
            algorithm_type: 算法类型
            
        Returns:
            算法配置
        """
        config = AlgorithmConfig(algorithm_type=algorithm_type)
        
        # 根据场景调整配置
        if scenario == DecisionScenario.HYPERPARAMETER_INIT:
            config.n_initial_points = 3
            config.epsilon = 0.15
            config.ucb_c = 2.5
            config.extra['strategy'] = 'ucb1'
            
        elif scenario == DecisionScenario.RESOURCE_ALLOCATION:
            config.population_size = 30
            config.mutation_rate = 0.15
            config.crossover_rate = 0.85
            config.selection_method = 'tournament'
            
        elif scenario == DecisionScenario.TRAINING_STRATEGY:
            config.learning_rate = 0.01
            config.discount_factor = 0.95
            config.epsilon = 0.2
            config.epsilon_decay = 0.98
            config.extra['method'] = 'q_learning'
            
        elif scenario == DecisionScenario.DATA_PREPROCESSING:
            config.acquisition_function = 'expected_improvement'
            config.exploration_weight = 0.1
            config.n_initial_points = 5
            
        elif scenario == DecisionScenario.MODEL_ARCHITECTURE:
            config.inference_depth = 3
            config.confidence_threshold = 0.6
            
        elif scenario == DecisionScenario.MODEL_SELECTION:
            config.ucb_c = 2.0
            config.extra['strategy'] = 'thompson'
        
        return config
    
    def _build_algorithm_context(self, context: DecisionContext,
                                tenant_id: str = None) -> AlgoContext:
        """构建算法上下文
        
        Args:
            context: 决策上下文
            tenant_id: 租户ID
            
        Returns:
            算法上下文
        """
        # 构建搜索空间
        search_space = self._build_search_space(context)
        
        # 构建历史观测
        observations = []
        for hist in context.history[-20:]:  # 最近20条
            observations.append({
                'action': hist.get('action', hist.get('params', {})),
                'reward': hist.get('reward', hist.get('score', 0.5)),
                'inputs': hist.get('inputs', {})
            })
        
        return AlgoContext(
            inputs=context.inputs,
            constraints=context.constraints,
            history=context.history,
            search_space=search_space,
            objective='maximize',
            objective_metric='score',
            observations=observations,
            metadata={
                'scenario': context.scenario.value,
                'tenant_id': tenant_id
            }
        )
    
    def _build_search_space(self, context: DecisionContext) -> Dict[str, Any]:
        """构建搜索空间
        
        Args:
            context: 决策上下文
            
        Returns:
            搜索空间定义
        """
        search_space = {}
        
        if context.scenario == DecisionScenario.HYPERPARAMETER_INIT:
            model_type = context.inputs.get('model_type', 'mlp')
            
            if model_type == 'transformer':
                search_space = {
                    'learning_rate': {'type': 'float', 'low': 1e-5, 'high': 5e-4},
                    'batch_size': {'type': 'categorical', 'choices': [8, 16, 32]},
                    'epochs': {'type': 'int', 'low': 3, 'high': 20},
                    'warmup_ratio': {'type': 'float', 'low': 0.0, 'high': 0.2}
                }
            else:
                search_space = {
                    'learning_rate': {'type': 'float', 'low': 1e-4, 'high': 1e-1},
                    'batch_size': {'type': 'categorical', 'choices': [16, 32, 64, 128]},
                    'epochs': {'type': 'int', 'low': 10, 'high': 100}
                }
                
        elif context.scenario == DecisionScenario.RESOURCE_ALLOCATION:
            search_space = {
                'cpu_cores': {'type': 'categorical', 'choices': [2, 4, 8, 16]},
                'memory_gb': {'type': 'categorical', 'choices': [8, 16, 32, 64]},
                'gpu_count': {'type': 'categorical', 'choices': [0, 1, 2, 4]},
                'gpu_memory_gb': {'type': 'categorical', 'choices': [8, 12, 16, 24]}
            }
            
        elif context.scenario == DecisionScenario.MODEL_ARCHITECTURE:
            data_type = context.inputs.get('data_type', 'tabular')
            
            if data_type == 'text':
                search_space = {
                    'architecture': {'type': 'categorical', 'choices': ['transformer', 'lstm', 'bert']},
                    'layers': {'type': 'categorical', 'choices': [6, 12, 24]},
                    'hidden_size': {'type': 'categorical', 'choices': [256, 512, 768]}
                }
            elif data_type == 'image':
                search_space = {
                    'architecture': {'type': 'categorical', 'choices': ['resnet', 'vgg', 'efficientnet']},
                    'layers': {'type': 'categorical', 'choices': [18, 34, 50, 101]},
                    'pretrained': {'type': 'categorical', 'choices': [True, False]}
                }
            else:
                search_space = {
                    'architecture': {'type': 'categorical', 'choices': ['mlp', 'xgboost', 'lightgbm']},
                    'layers': {'type': 'categorical', 'choices': [[64], [128, 64], [256, 128, 64]]}
                }
        
        return search_space
    
    def _fallback_decision(self, context: DecisionContext, 
                          tenant_id: str, user_id: str,
                          error_msg: str) -> DecisionResult:
        """回退决策 - 当算法失败时使用规则决策
        
        Args:
            context: 决策上下文
            tenant_id: 租户ID
            user_id: 用户ID
            error_msg: 错误信息
            
        Returns:
            决策结果
        """
        self.logger.warning(f"Falling back to rule-based decision: {error_msg}")
        
        # 使用规则决策
        recommended_action, confidence, reasoning, alternatives = self._execute_rule_based_decision(
            context, DecisionAlgorithm.RULE_BASED
        )
        
        decision_id = f"decision_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{str(uuid.uuid4())[:8]}"
        
        return DecisionResult(
            decision_id=decision_id,
            scenario=context.scenario.value,
            recommended_action=recommended_action,
            confidence=confidence * 0.8,  # 降低置信度
            reasoning=f"[回退决策] {reasoning}",
            alternatives=alternatives,
            execution_plan=self._generate_execution_plan(context, recommended_action),
            metadata={
                "algorithm": "rule_based_fallback",
                "timestamp": datetime.utcnow().isoformat(),
                "fallback_reason": error_msg
            }
        )
    
    def update_algorithm(self, scenario: DecisionScenario, action: Dict[str, Any],
                        reward: float, context: DecisionContext = None) -> None:
        """更新算法（用于强化学习反馈）
        
        Args:
            scenario: 决策场景
            action: 执行的动作
            reward: 获得的奖励
            context: 决策上下文
        """
        algorithm_type = self._scenario_algorithm_map.get(
            scenario, AlgorithmType.KNOWLEDGE_REASONING
        )
        cache_key = f"{scenario.value}:{algorithm_type.value}"
        
        if cache_key in self._algorithms:
            algo_context = None
            if context:
                algo_context = self._build_algorithm_context(context)
            
            self._algorithms[cache_key].update(action, reward, algo_context)
            self.logger.debug(f"Algorithm {cache_key} updated with reward={reward}")

    # ==========================================================================
    # Agent 高级推理
    # ==========================================================================

    def execute_agent_reasoning(self, task: AgentTask, 
                               tenant_id: str = None,
                               user_id: str = None,
                               mode: AgentMode = AgentMode.MULTI_STEP,
                               llm_client: Any = None) -> AgentExecutionResult:
        """执行 Agent 高级推理
        
        Args:
            task: Agent 任务
            tenant_id: 租户ID
            user_id: 用户ID
            mode: Agent 执行模式
            llm_client: LLM 客户端（可选）
            
        Returns:
            AgentExecutionResult: 执行结果
        """
        import time
        start_time = time.time()
        
        try:
            self.logger.info(f"开始 Agent 推理，任务: {task.task_id}, 模式: {mode.value}")
            
            # 获取或创建 Agent
            agent = self._get_or_create_agent(mode, task, llm_client)
            
            # 构建输入
            agent_input = self._build_agent_input(task)
            
            # 执行 Agent
            result_state = agent.invoke(agent_input, config={
                'max_iterations': task.max_iterations,
                'metadata': {
                    'tenant_id': tenant_id,
                    'user_id': user_id,
                    'task_id': task.task_id
                }
            })
            
            execution_time = time.time() - start_time
            
            # 构建执行结果
            execution_result = AgentExecutionResult(
                execution_id=f"agent_exec_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{str(uuid.uuid4())[:8]}",
                agent_type=mode.value,
                input_query=task.description,
                final_answer=result_state.final_answer or result_state.output,
                reasoning_steps=result_state.intermediate_steps,
                tool_calls=self._extract_tool_calls(result_state),
                iterations=result_state.iteration,
                status=result_state.status.value,
                confidence=self._calculate_confidence(result_state),
                execution_time=execution_time,
                metadata={
                    'task_id': task.task_id,
                    'task_type': task.task_type,
                    'mode': mode.value,
                    'messages_count': len(result_state.messages),
                    'plan': result_state.plan if hasattr(result_state, 'plan') else [],
                    'reflections': result_state.reflections if hasattr(result_state, 'reflections') else []
                }
            )
            
            # 持久化结果
            if tenant_id and user_id:
                self._save_agent_execution(tenant_id, user_id, task, execution_result)
            
            self.logger.info(f"Agent 推理完成，执行ID: {execution_result.execution_id}")
            return execution_result
            
        except Exception as e:
            self.logger.error(f"Agent 推理失败: {str(e)}")
            execution_time = time.time() - start_time
            
            return AgentExecutionResult(
                execution_id=f"agent_exec_error_{str(uuid.uuid4())[:8]}",
                agent_type=mode.value,
                input_query=task.description,
                final_answer=None,
                reasoning_steps=[],
                tool_calls=[],
                iterations=0,
                status="failed",
                confidence=0.0,
                execution_time=execution_time,
                metadata={'error': str(e)}
            )

    def _get_or_create_agent(self, mode: AgentMode, task: AgentTask, 
                            llm_client: Any = None) -> Any:
        """获取或创建 Agent 实例
        
        Args:
            mode: Agent 模式
            task: 任务定义
            llm_client: LLM 客户端
            
        Returns:
            Agent 实例
        """
        cache_key = f"{mode.value}:{task.task_type}"
        
        if cache_key not in self._agents:
            # 获取任务所需的工具
            tools = self._get_tools_for_task(task)
            
            # 创建系统提示
            system_prompt = self._create_agent_system_prompt(mode, task)
            
            # 创建 Agent
            if mode == AgentMode.MULTI_STEP:
                self._agents[cache_key] = create_react_agent(
                    name=f"react_{task.task_type}",
                    tools=tools,
                    llm_client=llm_client,
                    system_prompt=system_prompt,
                    max_iterations=task.max_iterations,
                    checkpointer=self._agent_checkpointer
                )
            elif mode == AgentMode.PLAN_EXECUTE:
                self._agents[cache_key] = create_plan_execute_agent(
                    name=f"plan_exec_{task.task_type}",
                    tools=tools,
                    llm_client=llm_client,
                    system_prompt=system_prompt,
                    max_iterations=task.max_iterations,
                    checkpointer=self._agent_checkpointer
                )
            elif mode == AgentMode.REFLEXION:
                config = AgentConfig(
                    name=f"reflexion_{task.task_type}",
                    max_iterations=task.max_iterations,
                    system_prompt=system_prompt
                )
                self._agents[cache_key] = ReflexionAgent(
                    config=config,
                    tools=tools,
                    llm_client=llm_client,
                    checkpointer=self._agent_checkpointer,
                    max_reflections=3
                )
            else:
                # 默认使用 ReAct
                self._agents[cache_key] = create_react_agent(
                    name=f"default_{task.task_type}",
                    tools=tools,
                    llm_client=llm_client,
                    system_prompt=system_prompt,
                    max_iterations=task.max_iterations,
                    checkpointer=self._agent_checkpointer
                )
            
            self.logger.info(f"Created Agent instance: {cache_key}")
        
        return self._agents[cache_key]

    def _get_tools_for_task(self, task: AgentTask) -> List[Tool]:
        """获取任务所需的工具
        
        Args:
            task: 任务定义
            
        Returns:
            工具列表
        """
        tools = []
        
        # 如果指定了工具，只获取指定的
        if task.required_tools:
            for tool_name in task.required_tools:
                tool = self._tool_registry.get(tool_name)
                if tool:
                    tools.append(tool)
        else:
            # 根据任务类型选择工具
            task_type_tools = {
                'data_analysis': ['analyze_data_characteristics', 'data_analyzer', 'calculator'],
                'model_recommendation': ['query_knowledge_base', 'recommend_hyperparameters', 'estimate_resources'],
                'hyperparameter_tuning': ['recommend_hyperparameters', 'evaluate_decision', 'calculator'],
                'resource_planning': ['estimate_resources', 'evaluate_decision'],
                'training_strategy': ['query_knowledge_base', 'recommend_hyperparameters', 'evaluate_decision'],
                'general': ['query_knowledge_base', 'calculator', 'current_time']
            }
            
            tool_names = task_type_tools.get(task.task_type, task_type_tools['general'])
            for tool_name in tool_names:
                tool = self._tool_registry.get(tool_name)
                if tool:
                    tools.append(tool)
        
        return tools

    def _create_agent_system_prompt(self, mode: AgentMode, task: AgentTask) -> str:
        """创建 Agent 系统提示
        
        Args:
            mode: Agent 模式
            task: 任务定义
            
        Returns:
            系统提示
        """
        base_prompt = f"""你是一个专业的 AI 决策助手，专注于机器学习和深度学习领域的决策支持。

当前任务类型: {task.task_type}
任务描述: {task.description}

"""
        
        if mode == AgentMode.MULTI_STEP:
            return base_prompt + """使用 ReAct 框架进行推理：
1. 思考 (Thought): 分析问题，思考需要什么信息
2. 行动 (Action): 调用合适的工具获取信息
3. 观察 (Observation): 分析工具返回的结果
4. 重复直到得出结论

完成推理后，请给出清晰的最终答案，标注"最终答案"。"""

        elif mode == AgentMode.PLAN_EXECUTE:
            return base_prompt + """使用计划执行模式：
1. 首先分析任务，制定详细的执行计划
2. 按计划逐步执行每个步骤
3. 使用工具获取必要信息
4. 评估执行结果，必要时调整计划
5. 汇总所有结果给出最终答案"""

        elif mode == AgentMode.REFLEXION:
            return base_prompt + """使用反思模式：
1. 生成初始分析和建议
2. 评估自己的分析是否全面、准确
3. 如果发现不足，进行反思并改进
4. 重复直到满意
5. 给出经过验证的最终建议"""

        return base_prompt + """请仔细分析任务，使用可用的工具收集信息，给出专业的建议。"""

    def _build_agent_input(self, task: AgentTask) -> str:
        """构建 Agent 输入
        
        Args:
            task: 任务定义
            
        Returns:
            输入字符串
        """
        input_parts = [f"任务: {task.description}"]
        
        if task.inputs:
            input_parts.append(f"\n输入参数:\n{json.dumps(task.inputs, ensure_ascii=False, indent=2)}")
        
        if task.constraints:
            input_parts.append(f"\n约束条件:\n{json.dumps(task.constraints, ensure_ascii=False, indent=2)}")
        
        return "\n".join(input_parts)

    def _extract_tool_calls(self, state: AgentState) -> List[Dict[str, Any]]:
        """从状态中提取工具调用记录
        
        Args:
            state: Agent 状态
            
        Returns:
            工具调用列表
        """
        tool_calls = []
        
        for step in state.intermediate_steps:
            action = step.get('action', {})
            if 'tool' in action:
                tool_calls.append({
                    'tool': action['tool'],
                    'arguments': action.get('arguments', {}),
                    'result': step.get('observation'),
                    'iteration': step.get('iteration', 0)
                })
        
        return tool_calls

    def _calculate_confidence(self, state: AgentState) -> float:
        """计算执行置信度
        
        Args:
            state: Agent 状态
            
        Returns:
            置信度 (0-1)
        """
        confidence = 0.5
        
        # 有最终答案加分
        if state.final_answer:
            confidence += 0.2
        
        # 执行了工具调用加分
        if state.intermediate_steps:
            confidence += min(len(state.intermediate_steps) * 0.05, 0.15)
        
        # 迭代次数适中加分
        if 2 <= state.iteration <= 5:
            confidence += 0.1
        elif state.iteration > 5:
            confidence += 0.05
        
        # 状态完成加分
        if state.status == AgentStatus.COMPLETED:
            confidence += 0.05
        
        return min(confidence, 0.95)

    def _save_agent_execution(self, tenant_id: str, user_id: str,
                             task: AgentTask, result: AgentExecutionResult):
        """保存 Agent 执行记录
        
        Args:
            tenant_id: 租户ID
            user_id: 用户ID
            task: 任务
            result: 执行结果
        """
        try:
            self.decision_repo.create({
                'tenant_id': tenant_id,
                'user_id': user_id,
                'decision_id': result.execution_id,
                'scenario': task.task_type,
                'algorithm': result.agent_type,
                'inputs': task.inputs,
                'constraints': task.constraints,
                'history_context': [],
                'recommended_action': {'answer': result.final_answer},
                'confidence': result.confidence,
                'reasoning': json.dumps(result.reasoning_steps, ensure_ascii=False),
                'alternatives': [],
                'execution_plan': {
                    'tool_calls': result.tool_calls,
                    'iterations': result.iterations,
                    'execution_time': result.execution_time
                }
            })
        except Exception as e:
            self.logger.error(f"Failed to save agent execution: {e}")

    def execute_complex_reasoning(self, query: str,
                                 context: Dict[str, Any] = None,
                                 tenant_id: str = None,
                                 user_id: str = None,
                                 mode: AgentMode = AgentMode.MULTI_STEP,
                                 tools: List[str] = None,
                                 llm_client: Any = None) -> AgentExecutionResult:
        """执行复杂推理（简化接口）
        
        Args:
            query: 查询问题
            context: 上下文信息
            tenant_id: 租户ID
            user_id: 用户ID
            mode: 执行模式
            tools: 指定使用的工具
            llm_client: LLM 客户端
            
        Returns:
            执行结果
        """
        # 推断任务类型
        task_type = self._infer_task_type(query)
        
        # 创建任务
        task = AgentTask(
            task_id=f"task_{str(uuid.uuid4())[:8]}",
            task_type=task_type,
            description=query,
            inputs=context or {},
            required_tools=tools or [],
            max_iterations=10
        )
        
        return self.execute_agent_reasoning(
            task=task,
            tenant_id=tenant_id,
            user_id=user_id,
            mode=mode,
            llm_client=llm_client
        )

    def _infer_task_type(self, query: str) -> str:
        """推断任务类型
        
        Args:
            query: 查询问题
            
        Returns:
            任务类型
        """
        query_lower = query.lower()
        
        if any(k in query_lower for k in ['数据', 'data', '分析', 'analysis']):
            return 'data_analysis'
        elif any(k in query_lower for k in ['模型', 'model', '架构', 'architecture']):
            return 'model_recommendation'
        elif any(k in query_lower for k in ['超参数', 'hyperparameter', '参数']):
            return 'hyperparameter_tuning'
        elif any(k in query_lower for k in ['资源', 'resource', 'gpu', 'cpu']):
            return 'resource_planning'
        elif any(k in query_lower for k in ['训练', 'training', '策略', 'strategy']):
            return 'training_strategy'
        else:
            return 'general'

    def create_multi_agent_workflow(self, task: AgentTask,
                                   agent_roles: List[Dict[str, Any]],
                                   tenant_id: str = None,
                                   user_id: str = None,
                                   llm_client: Any = None) -> AgentExecutionResult:
        """创建多 Agent 协作工作流
        
        Args:
            task: 任务定义
            agent_roles: Agent 角色定义列表
            tenant_id: 租户ID
            user_id: 用户ID
            llm_client: LLM 客户端
            
        Returns:
            执行结果
        """
        import time
        start_time = time.time()
        
        try:
            self.logger.info(f"创建多 Agent 工作流，任务: {task.task_id}")
            
            # 创建角色
            roles = []
            for role_config in agent_roles:
                tools = self._get_tools_for_task(AgentTask(
                    task_id=role_config.get('name', 'default'),
                    task_type=role_config.get('specialty', 'general'),
                    description=role_config.get('description', ''),
                    inputs={}
                ))
                
                roles.append(AgentRole(
                    name=role_config['name'],
                    description=role_config.get('description', ''),
                    system_prompt=role_config.get('system_prompt', f"你是 {role_config['name']}"),
                    tools=tools
                ))
            
            # 创建多 Agent 系统
            config = AgentConfig(
                name=f"multi_agent_{task.task_id}",
                max_iterations=task.max_iterations
            )
            
            multi_agent = MultiAgentSystem(
                config=config,
                roles=roles,
                llm_client=llm_client,
                checkpointer=self._agent_checkpointer,
                collaboration_mode="supervisor"
            )
            
            # 执行
            result_state = multi_agent.invoke(self._build_agent_input(task))
            
            execution_time = time.time() - start_time
            
            return AgentExecutionResult(
                execution_id=f"multi_agent_{str(uuid.uuid4())[:8]}",
                agent_type="multi_agent",
                input_query=task.description,
                final_answer=result_state.final_answer or result_state.output,
                reasoning_steps=result_state.intermediate_steps,
                tool_calls=self._extract_tool_calls(result_state),
                iterations=result_state.iteration,
                status=result_state.status.value,
                confidence=self._calculate_confidence(result_state),
                execution_time=execution_time,
                metadata={
                    'roles': [r.name for r in roles],
                    'collaboration_mode': 'supervisor'
                }
            )
            
        except Exception as e:
            self.logger.error(f"多 Agent 工作流失败: {str(e)}")
            return AgentExecutionResult(
                execution_id=f"multi_agent_error_{str(uuid.uuid4())[:8]}",
                agent_type="multi_agent",
                input_query=task.description,
                final_answer=None,
                reasoning_steps=[],
                tool_calls=[],
                iterations=0,
                status="failed",
                confidence=0.0,
                execution_time=time.time() - start_time,
                metadata={'error': str(e)}
            )

    def stream_agent_reasoning(self, task: AgentTask,
                              tenant_id: str = None,
                              user_id: str = None,
                              mode: AgentMode = AgentMode.MULTI_STEP,
                              llm_client: Any = None):
        """流式执行 Agent 推理
        
        Args:
            task: 任务定义
            tenant_id: 租户ID
            user_id: 用户ID
            mode: 执行模式
            llm_client: LLM 客户端
            
        Yields:
            每一步的执行状态
        """
        try:
            self.logger.info(f"开始流式 Agent 推理，任务: {task.task_id}")
            
            agent = self._get_or_create_agent(mode, task, llm_client)
            agent_input = self._build_agent_input(task)
            
            for chunk in agent.stream(agent_input, config={
                'max_iterations': task.max_iterations,
                'metadata': {'tenant_id': tenant_id, 'user_id': user_id}
            }):
                yield {
                    'node': chunk['node'],
                    'status': chunk['state'].status.value,
                    'iteration': chunk['state'].iteration,
                    'messages_count': len(chunk['state'].messages),
                    'has_tool_calls': len(chunk['state'].pending_tool_calls) > 0,
                    'intermediate_steps': len(chunk['state'].intermediate_steps),
                    'final_answer': chunk['state'].final_answer
                }
                
        except Exception as e:
            self.logger.error(f"流式推理失败: {str(e)}")
            yield {'error': str(e), 'status': 'failed'}

    def get_available_tools(self) -> List[Dict[str, Any]]:
        """获取可用工具列表
        
        Returns:
            工具信息列表
        """
        tools = self._tool_registry.list_tools()
        return [
            {
                'name': t.name,
                'description': t.description,
                'category': t.category.value,
                'parameters': [
                    {
                        'name': p.name,
                        'type': p.type,
                        'description': p.description,
                        'required': p.required
                    }
                    for p in t.parameters
                ]
            }
            for t in tools
        ]

    def register_custom_tool(self, name: str, func: Callable,
                            description: str = None,
                            category: str = "custom") -> bool:
        """注册自定义工具
        
        Args:
            name: 工具名称
            func: 工具函数
            description: 描述
            category: 类别
            
        Returns:
            是否成功
        """
        try:
            tool_obj = Tool(
                name=name,
                description=description or func.__doc__ or f"执行 {name}",
                func=func,
                category=ToolCategory.CUSTOM
            )
            self._tool_registry.register(tool_obj)
            self.logger.info(f"Registered custom tool: {name}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to register tool: {e}")
            return False

    def _save_decision(self, tenant_id: str, user_id: str, 
                      context: DecisionContext, result: DecisionResult,
                      algorithm: AlgorithmType):
        """保存决策记录到数据库"""
        try:
            self.decision_repo.create({
                'tenant_id': tenant_id,
                'user_id': user_id,
                'decision_id': result.decision_id,
                'scenario': result.scenario,
                'algorithm': algorithm.value,
                'inputs': context.inputs,
                'constraints': context.constraints,
                'history_context': context.history,
                'recommended_action': result.recommended_action,
                'confidence': result.confidence,
                'reasoning': result.reasoning,
                'alternatives': result.alternatives,
                'execution_plan': result.execution_plan
            })
        except Exception as e:
            self.logger.error(f"Failed to save decision: {e}")

    def get_decision(self, decision_id: str, tenant_id: str) -> Optional[Dict[str, Any]]:
        """获取决策详情
        
        Args:
            decision_id: 决策ID
            tenant_id: 租户ID
            
        Returns:
            决策详情
        """
        try:
            decision = self.decision_repo.get_by_decision_id(decision_id)
            if not decision:
                return None
            
            # 验证租户权限
            d_tenant = decision.tenant_id if hasattr(decision, 'tenant_id') else decision.get('tenant_id')
            if d_tenant != tenant_id:
                return None
            
            return self._decision_to_dict(decision)
        except Exception as e:
            self.logger.error(f"Failed to get decision: {e}")
            return None

    def list_decisions(self, tenant_id: str, user_id: Optional[str] = None,
                      scenario: Optional[str] = None,
                      page: int = 1, page_size: int = 20) -> Dict[str, Any]:
        """列出决策记录
        
        Args:
            tenant_id: 租户ID
            user_id: 用户ID过滤
            scenario: 场景过滤
            page: 页码
            page_size: 每页数量
            
        Returns:
            决策列表和分页信息
        """
        try:
            offset = (page - 1) * page_size
            decisions, total = self.decision_repo.list_by_tenant(
                tenant_id=tenant_id,
                user_id=user_id,
                scenario=scenario,
                limit=page_size,
                offset=offset
            )
            
            return {
                'items': [self._decision_to_dict(d) for d in decisions],
                'total': total,
                'page': page,
                'page_size': page_size,
                'total_pages': (total + page_size - 1) // page_size
            }
        except Exception as e:
            self.logger.error(f"Failed to list decisions: {e}")
            return {'items': [], 'total': 0, 'page': page, 'page_size': page_size, 'total_pages': 0}

    def provide_feedback(self, decision_id: str, tenant_id: str, user_id: str,
                        feedback_score: float, feedback_comment: Optional[str] = None) -> bool:
        """提供决策反馈
        
        Args:
            decision_id: 决策ID
            tenant_id: 租户ID
            user_id: 用户ID
            feedback_score: 反馈评分 (0-1)
            feedback_comment: 反馈说明
            
        Returns:
            是否成功
        """
        try:
            decision = self.decision_repo.get_by_decision_id(decision_id)
            if not decision:
                return False
            
            d_id = decision.id if hasattr(decision, 'id') else decision.get('id')
            
            self.decision_repo.update(d_id, {
                'feedback_score': feedback_score,
                'feedback_comment': feedback_comment
            })
            
            # 记录经验
            self._record_experience_from_feedback(
                tenant_id, user_id, decision, feedback_score, feedback_comment
            )
            
            return True
        except Exception as e:
            self.logger.error(f"Failed to provide feedback: {e}")
            return False

    def _record_experience_from_feedback(self, tenant_id: str, user_id: str,
                                        decision, feedback_score: float,
                                        feedback_comment: Optional[str]):
        """从反馈中记录经验"""
        try:
            d_dict = self._decision_to_dict(decision)
            
            self.experience_repo.create({
                'tenant_id': tenant_id,
                'user_id': user_id,
                'experience_type': 'decision_feedback',
                'scenario': d_dict.get('scenario'),
                'context': d_dict.get('inputs'),
                'action': d_dict.get('recommended_action'),
                'result': {'feedback_score': feedback_score, 'comment': feedback_comment},
                'reward': feedback_score,
                'decision_id': d_dict.get('decision_id'),
                'effectiveness': feedback_score,
                'is_positive': feedback_score >= 0.5
            })
        except Exception as e:
            self.logger.error(f"Failed to record experience: {e}")

    def get_decision_statistics(self, tenant_id: str, 
                               user_id: Optional[str] = None) -> Dict[str, Any]:
        """获取决策统计信息
        
        Args:
            tenant_id: 租户ID
            user_id: 用户ID过滤
            
        Returns:
            统计信息
        """
        try:
            return self.decision_repo.get_statistics(tenant_id, user_id)
        except Exception as e:
            self.logger.error(f"Failed to get statistics: {e}")
            return {'total': 0, 'by_scenario': {}, 'by_algorithm': {}, 
                   'avg_confidence': 0, 'avg_feedback_score': 0}

    # ==========================================================================
    # 自适应优化
    # ==========================================================================

    def adaptive_optimization(self, config: AdaptiveConfiguration,
                            tenant_id: str = None, 
                            user_id: str = None) -> AdaptiveOptimizationResult:
        """
        自适应优化
        
        Args:
            config: 自适应配置
            tenant_id: 租户ID
            user_id: 用户ID
            
        Returns:
            AdaptiveOptimizationResult: 优化结果
        """
        try:
            self.logger.info(f"开始自适应优化，参数: {config.parameter_name}")
            
            # 生成优化ID
            optimization_id = f"optimization_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{str(uuid.uuid4())[:8]}"
            
            # 执行自适应优化
            original_value = config.current_value
            optimized_value, improvement_metric, improvement_value, adjustment_reason = \
                self._execute_adaptive_optimization(config)
            
            # 创建优化结果
            result = AdaptiveOptimizationResult(
                optimization_id=optimization_id,
                parameter_name=config.parameter_name,
                original_value=original_value,
                optimized_value=optimized_value,
                improvement_metric=improvement_metric,
                improvement_value=improvement_value,
                adjustment_reason=adjustment_reason,
                timestamp=datetime.utcnow()
            )
            
            # 持久化优化记录
            if tenant_id and user_id:
                self._save_adaptive_optimization(tenant_id, user_id, config, result)
            
            self.logger.info(f"自适应优化完成，优化ID: {optimization_id}")
            return result
            
        except Exception as e:
            self.logger.error(f"自适应优化失败: {str(e)}")
            raise

    def _save_adaptive_optimization(self, tenant_id: str, user_id: str,
                                   config: AdaptiveConfiguration,
                                   result: AdaptiveOptimizationResult):
        """保存自适应优化记录"""
        try:
            self.adaptive_opt_repo.create({
                'tenant_id': tenant_id,
                'user_id': user_id,
                'optimization_id': result.optimization_id,
                'parameter_name': result.parameter_name,
                'adjustment_strategy': config.adjustment_strategy,
                'adjustment_range': config.adjustment_range,
                'monitoring_metrics': config.monitoring_metrics,
                'original_value': result.original_value,
                'optimized_value': result.optimized_value,
                'improvement_metric': result.improvement_metric,
                'improvement_value': result.improvement_value,
                'adjustment_reason': result.adjustment_reason
            })
        except Exception as e:
            self.logger.error(f"Failed to save adaptive optimization: {e}")

    def list_adaptive_optimizations(self, tenant_id: str, user_id: Optional[str] = None,
                                   parameter_name: Optional[str] = None,
                                   page: int = 1, page_size: int = 20) -> Dict[str, Any]:
        """列出自适应优化记录
        
        Args:
            tenant_id: 租户ID
            user_id: 用户ID过滤
            parameter_name: 参数名过滤
            page: 页码
            page_size: 每页数量
            
        Returns:
            优化列表和分页信息
        """
        try:
            offset = (page - 1) * page_size
            opts, total = self.adaptive_opt_repo.list_by_tenant(
                tenant_id=tenant_id,
                user_id=user_id,
                parameter_name=parameter_name,
                limit=page_size,
                offset=offset
            )
            
            return {
                'items': [self._adaptive_opt_to_dict(o) for o in opts],
                'total': total,
                'page': page,
                'page_size': page_size,
                'total_pages': (total + page_size - 1) // page_size
            }
        except Exception as e:
            self.logger.error(f"Failed to list adaptive optimizations: {e}")
            return {'items': [], 'total': 0, 'page': page, 'page_size': page_size, 'total_pages': 0}

    def apply_optimization(self, optimization_id: str, tenant_id: str) -> bool:
        """应用优化结果
        
        Args:
            optimization_id: 优化ID
            tenant_id: 租户ID
            
        Returns:
            是否成功
        """
        try:
            # 查找优化记录
            opts, _ = self.adaptive_opt_repo.list_by_tenant(tenant_id, limit=1000)
            opt = None
            for o in opts:
                o_id = o.optimization_id if hasattr(o, 'optimization_id') else o.get('optimization_id')
                if o_id == optimization_id:
                    opt = o
                    break
            
            if not opt:
                return False
            
            o_id = opt.id if hasattr(opt, 'id') else opt.get('id')
            self.adaptive_opt_repo.update(o_id, {
                'applied': True,
                'applied_at': datetime.now()
            })
            
            return True
        except Exception as e:
            self.logger.error(f"Failed to apply optimization: {e}")
            return False

    # ==========================================================================
    # 知识库管理
    # ==========================================================================

    def update_knowledge_base(self, knowledge: Dict[str, Any],
                            tenant_id: str = None, 
                            user_id: str = None) -> bool:
        """
        更新知识库
        
        Args:
            knowledge: 知识数据
            tenant_id: 租户ID
            user_id: 用户ID
            
        Returns:
            bool: 是否成功更新
        """
        try:
            self.logger.info("更新知识库")
            
            # 更新内存知识库（兼容旧代码）
            self.knowledge_base.update(knowledge)
            
            # 持久化到数据库
            if tenant_id and user_id:
                self.knowledge_repo.create({
                    'tenant_id': tenant_id,
                    'user_id': user_id,
                    'knowledge_type': knowledge.get('type', 'general'),
                    'category': knowledge.get('category', ''),
                    'title': knowledge.get('title', '未命名知识'),
                    'content': knowledge,
                    'source': knowledge.get('source', 'user_input'),
                    'is_public': knowledge.get('is_public', False)
                })
            
            self.logger.info("知识库更新完成")
            return True
            
        except Exception as e:
            self.logger.error(f"更新知识库失败: {str(e)}")
            return False

    def get_knowledge_graph(self, query: str, tenant_id: str = None) -> Dict[str, Any]:
        """
        获取知识图谱
        
        Args:
            query: 查询语句
            tenant_id: 租户ID
            
        Returns:
            Dict[str, Any]: 知识图谱数据
        """
        try:
            self.logger.info(f"查询知识图谱: {query}")
            
            # 搜索相关知识
            if tenant_id:
                related_knowledge = self.knowledge_repo.search(
                    tenant_id, query, limit=10
                )
            else:
                related_knowledge = []
            
            # 构建知识图谱
            if related_knowledge:
                knowledge_graph = self._build_knowledge_graph_from_db(query, related_knowledge)
            else:
                # 使用规则推理
                knowledge_graph = self._rule_based_knowledge_inference(query)
            
            self.logger.info("知识图谱查询完成")
            return knowledge_graph
            
        except Exception as e:
            self.logger.error(f"查询知识图谱失败: {str(e)}")
            raise

    def _build_knowledge_graph_from_db(self, query: str, 
                                      knowledge_list: List[Any]) -> Dict[str, Any]:
        """从数据库知识构建知识图谱"""
        entities = []
        relationships = []
        recommendations = []
        
        for k in knowledge_list:
            k_dict = self._knowledge_to_dict(k)
            
            # 添加实体
            entities.append({
                'id': k_dict.get('id'),
                'type': k_dict.get('knowledge_type'),
                'name': k_dict.get('title'),
                'content': k_dict.get('content')
            })
            
            # 添加关系
            related = k_dict.get('related_entities', [])
            for r in related:
                relationships.append({
                    'source': k_dict.get('id'),
                    'target': r.get('id'),
                    'relation': r.get('relation', 'related_to')
                })
            
            # 提取推荐
            content = k_dict.get('content', {})
            if isinstance(content, dict):
                if 'best_practices' in content:
                    recommendations.extend(content['best_practices'])
                if 'recommendations' in content:
                    recommendations.extend(content['recommendations'])
        
        return {
            'query': query,
            'entities': entities,
            'relationships': relationships,
            'recommendations': list(set(recommendations))[:5],
            'inference_type': 'knowledge_based'
        }

    def list_knowledge(self, tenant_id: str, knowledge_type: Optional[str] = None,
                      category: Optional[str] = None,
                      page: int = 1, page_size: int = 20) -> Dict[str, Any]:
        """列出知识
        
        Args:
            tenant_id: 租户ID
            knowledge_type: 类型过滤
            category: 类别过滤
            page: 页码
            page_size: 每页数量
            
        Returns:
            知识列表和分页信息
        """
        try:
            offset = (page - 1) * page_size
            knowledge_list, total = self.knowledge_repo.list_available(
                tenant_id=tenant_id,
                knowledge_type=knowledge_type,
                category=category,
                limit=page_size,
                offset=offset
            )
            
            return {
                'items': [self._knowledge_to_dict(k) for k in knowledge_list],
                'total': total,
                'page': page,
                'page_size': page_size,
                'total_pages': (total + page_size - 1) // page_size
            }
        except Exception as e:
            self.logger.error(f"Failed to list knowledge: {e}")
            return {'items': [], 'total': 0, 'page': page, 'page_size': page_size, 'total_pages': 0}

    def create_knowledge(self, tenant_id: str, user_id: str,
                        data: Dict[str, Any]) -> Dict[str, Any]:
        """创建知识
        
        Args:
            tenant_id: 租户ID
            user_id: 用户ID
            data: 知识数据
            
        Returns:
            创建的知识
        """
        try:
            knowledge = self.knowledge_repo.create({
                'tenant_id': tenant_id,
                'user_id': user_id,
                'knowledge_type': data.get('knowledge_type', 'general'),
                'category': data.get('category'),
                'title': data.get('title'),
                'content': data.get('content'),
                'related_entities': data.get('related_entities', []),
                'relationships': data.get('relationships', []),
                'source': data.get('source', 'user_input'),
                'confidence': data.get('confidence', 1.0),
                'is_public': data.get('is_public', False),
                'tags': data.get('tags', [])
            })
            return self._knowledge_to_dict(knowledge)
        except Exception as e:
            self.logger.error(f"Failed to create knowledge: {e}")
            raise

    # ==========================================================================
    # 经验积累
    # ==========================================================================

    def experience_accumulation(self, experience_data: Dict[str, Any],
                               tenant_id: str = None, 
                               user_id: str = None) -> bool:
        """
        经验积累
        
        Args:
            experience_data: 经验数据
            tenant_id: 租户ID
            user_id: 用户ID
            
        Returns:
            bool: 是否成功积累
        """
        try:
            self.logger.info("积累经验数据")
            
            if tenant_id and user_id:
                self.experience_repo.create({
                    'tenant_id': tenant_id,
                    'user_id': user_id,
                    'experience_type': experience_data.get('type', 'general'),
                    'scenario': experience_data.get('scenario'),
                    'context': experience_data.get('context', {}),
                    'action': experience_data.get('action', {}),
                    'result': experience_data.get('result', {}),
                    'reward': experience_data.get('reward', 0.0),
                    'decision_id': experience_data.get('decision_id'),
                    'effectiveness': experience_data.get('effectiveness'),
                    'lessons_learned': experience_data.get('lessons_learned'),
                    'is_positive': experience_data.get('is_positive', True),
                    'tags': experience_data.get('tags', [])
                })
            
            self.logger.info("经验积累完成")
            return True
            
        except Exception as e:
            self.logger.error(f"经验积累失败: {str(e)}")
            return False

    def list_experiences(self, tenant_id: str, user_id: Optional[str] = None,
                        scenario: Optional[str] = None,
                        is_positive: Optional[bool] = None,
                        page: int = 1, page_size: int = 20) -> Dict[str, Any]:
        """列出经验记录
        
        Args:
            tenant_id: 租户ID
            user_id: 用户ID过滤
            scenario: 场景过滤
            is_positive: 是否正面经验
            page: 页码
            page_size: 每页数量
            
        Returns:
            经验列表和分页信息
        """
        try:
            offset = (page - 1) * page_size
            experiences, total = self.experience_repo.list_by_tenant(
                tenant_id=tenant_id,
                user_id=user_id,
                scenario=scenario,
                is_positive=is_positive,
                limit=page_size,
                offset=offset
            )
            
            return {
                'items': [self._experience_to_dict(e) for e in experiences],
                'total': total,
                'page': page,
                'page_size': page_size,
                'total_pages': (total + page_size - 1) // page_size
            }
        except Exception as e:
            self.logger.error(f"Failed to list experiences: {e}")
            return {'items': [], 'total': 0, 'page': page, 'page_size': page_size, 'total_pages': 0}

    # ==========================================================================
    # 内部方法
    # ==========================================================================

    def _query_related_knowledge(self, tenant_id: str, scenario: str) -> List[Any]:
        """查询相关知识"""
        try:
            knowledge_list, _ = self.knowledge_repo.list_available(
                tenant_id, category=scenario, limit=5
            )
            return knowledge_list
        except Exception as e:
            self.logger.warning(f"Failed to query related knowledge: {e}")
            return []

    def _select_decision_algorithm(self, scenario: DecisionScenario) -> AlgorithmType:
        """选择决策算法类型
        
        Args:
            scenario: 决策场景
            
        Returns:
            算法类型
        """
        return self._scenario_algorithm_map.get(scenario, AlgorithmType.KNOWLEDGE_REASONING)

    def _execute_rule_based_decision(self, context: DecisionContext, 
                                   algorithm: DecisionAlgorithm = None) -> tuple:
        """执行基于规则的决策"""
        inputs = context.inputs
        constraints = context.constraints
        
        if context.scenario == DecisionScenario.DATA_PREPROCESSING:
            data_type = inputs.get('data_type', 'numerical')
            
            if data_type == 'text':
                recommended_action = {
                    "strategy": "text_preprocessing",
                    "parameters": {"tokenization": "bert_tokenizer", "max_length": 512, "padding": True}
                }
                confidence = 0.88
                reasoning = "文本数据推荐使用BERT分词器进行预处理"
            elif data_type == 'image':
                recommended_action = {
                    "strategy": "image_preprocessing",
                    "parameters": {"resize": [224, 224], "normalization": "imagenet", "augmentation": True}
                }
                confidence = 0.85
                reasoning = "图像数据推荐使用ImageNet标准化预处理"
            else:
                recommended_action = {
                    "strategy": "standardization",
                    "parameters": {"normalization_type": "z-score", "handling_missing": "mean_imputation"}
                }
                confidence = 0.82
                reasoning = "数值数据推荐使用Z-score标准化处理"
                
            alternatives = [
                {"strategy": "normalization", "confidence": 0.75, "reasoning": "归一化适用于数值范围固定的特征"},
                {"strategy": "robust_scaling", "confidence": 0.70, "reasoning": "鲁棒缩放适用于包含异常值的数据"}
            ]
            
        elif context.scenario == DecisionScenario.MODEL_ARCHITECTURE:
            task_type = inputs.get('task_type', 'classification')
            data_type = inputs.get('data_type', 'tabular')
            data_size = inputs.get('data_size', 'medium')
            
            if data_type == 'text':
                recommended_action = {
                    "architecture": "transformer",
                    "layers": 6 if data_size == 'small' else 12,
                    "attention_heads": 8,
                    "hidden_size": 512 if data_size == 'small' else 768
                }
                confidence = 0.92
                reasoning = "文本任务推荐使用Transformer架构"
            elif data_type == 'image':
                recommended_action = {
                    "architecture": "resnet",
                    "layers": 18 if data_size == 'small' else 50,
                    "pretrained": True
                }
                confidence = 0.89
                reasoning = "图像任务推荐使用ResNet架构"
            else:
                recommended_action = {
                    "architecture": "mlp",
                    "layers": [128, 64, 32],
                    "activation": "relu",
                    "dropout": 0.2
                }
                confidence = 0.80
                reasoning = "表格数据推荐使用多层感知机"
                
            alternatives = [
                {"architecture": "lstm" if data_type == 'text' else "cnn", 
                 "confidence": 0.75, "reasoning": "备选架构适用于特定场景"}
            ]
            
        elif context.scenario == DecisionScenario.HYPERPARAMETER_INIT:
            model_type = inputs.get('model_type', 'transformer')
            data_size = inputs.get('data_size', 'medium')
            
            if model_type == 'transformer':
                learning_rate = 2e-5 if data_size == 'large' else 1e-4
                batch_size = 16 if data_size == 'large' else 32
                epochs = 3 if data_size == 'large' else 10
            else:
                learning_rate = 1e-3
                batch_size = 64 if data_size == 'small' else 32
                epochs = 50 if data_size == 'small' else 100
                
            recommended_action = {
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "epochs": epochs,
                "optimizer": "adamw" if model_type == 'transformer' else "adam"
            }
            confidence = 0.85
            reasoning = f"基于{model_type}模型和{data_size}数据规模的经验推荐"
            
            alternatives = [
                {"learning_rate": learning_rate * 2, "batch_size": batch_size // 2,
                 "epochs": epochs // 2, "confidence": 0.78, "reasoning": "更激进的参数设置"}
            ]
            
        else:  # RESOURCE_ALLOCATION 和其他
            model_complexity = inputs.get('model_complexity', 'medium')
            data_size = inputs.get('data_size', 'medium')
            
            if model_complexity == 'high' or data_size == 'large':
                recommended_action = {"cpu_cores": 8, "memory_gb": 32, "gpu_count": 2, "gpu_memory_gb": 16}
                confidence = 0.88
                reasoning = "高复杂度模型或大数据集需要更多计算资源"
            elif model_complexity == 'low' or data_size == 'small':
                recommended_action = {"cpu_cores": 2, "memory_gb": 8, "gpu_count": 1, "gpu_memory_gb": 8}
                confidence = 0.82
                reasoning = "低复杂度模型或小数据集使用基础资源配置"
            else:
                recommended_action = {"cpu_cores": 4, "memory_gb": 16, "gpu_count": 1, "gpu_memory_gb": 12}
                confidence = 0.85
                reasoning = "中等复杂度模型使用标准资源配置"
                
            alternatives = [
                {"cpu_cores": recommended_action["cpu_cores"] * 2,
                    "memory_gb": recommended_action["memory_gb"] * 2,
                 "confidence": 0.75, "reasoning": "更多资源配置，适用于性能优先场景"}
            ]
            
        return recommended_action, confidence, reasoning, alternatives
        
    def _generate_execution_plan(self, context: DecisionContext, action: Any) -> Dict[str, Any]:
        """生成执行计划"""
        execution_plan = {
            "steps": [],
            "estimated_time": "30 minutes",
            "resource_requirements": {},
            "rollback_plan": {}
        }
        
        if context.scenario == DecisionScenario.DATA_PREPROCESSING:
            execution_plan["steps"] = ["加载原始数据集", "执行数据清洗", "应用标准化处理", "保存预处理后的数据"]
            execution_plan["estimated_time"] = "15 minutes"
        elif context.scenario == DecisionScenario.MODEL_ARCHITECTURE:
            execution_plan["steps"] = ["初始化模型架构", "配置模型参数", "构建计算图", "验证模型结构"]
            execution_plan["estimated_time"] = "10 minutes"
        elif context.scenario == DecisionScenario.HYPERPARAMETER_INIT:
            execution_plan["steps"] = ["设置超参数配置", "初始化优化器", "配置学习率调度器", "验证配置正确性"]
            execution_plan["estimated_time"] = "5 minutes"
        else:
            execution_plan["steps"] = ["申请计算资源", "配置环境变量", "启动训练实例", "验证资源配置"]
            execution_plan["estimated_time"] = "5 minutes"
            
        return execution_plan

    def _execute_adaptive_optimization(self, config: AdaptiveConfiguration) -> tuple:
        """执行自适应优化"""
        parameter_name = config.parameter_name
        current_value = config.current_value
        strategy = config.adjustment_strategy
        metrics = config.monitoring_metrics
        
        if parameter_name == "learning_rate":
            if strategy == "reduce_on_plateau":
                optimized_value = current_value * 0.5
                improvement_metric = "convergence_speed"
                improvement_value = 0.20
                adjustment_reason = "检测到训练停滞，降低学习率以提高收敛性"
            elif strategy == "cosine_annealing":
                optimized_value = current_value * 0.8
                improvement_metric = "training_stability"
                improvement_value = 0.15
                adjustment_reason = "使用余弦退火策略调整学习率"
            else:
                optimized_value = current_value * 0.9
                improvement_metric = "loss_reduction"
                improvement_value = 0.12
                adjustment_reason = "基于梯度变化趋势的自适应调整"
                
        elif parameter_name == "batch_size":
            if "memory_usage" in metrics:
                optimized_value = min(current_value * 2, 128)
                improvement_metric = "training_speed"
                improvement_value = 0.25
                adjustment_reason = "基于内存使用情况优化批次大小"
            else:
                optimized_value = current_value
                improvement_metric = "stability"
                improvement_value = 0.05
                adjustment_reason = "保持当前批次大小以确保稳定性"
                
        elif parameter_name == "dropout_rate":
            if "overfitting" in metrics:
                optimized_value = min(current_value + 0.1, 0.5)
                improvement_metric = "generalization"
                improvement_value = 0.18
                adjustment_reason = "检测到过拟合，增加dropout率"
            elif "underfitting" in metrics:
                optimized_value = max(current_value - 0.1, 0.0)
                improvement_metric = "model_capacity"
                improvement_value = 0.10
                adjustment_reason = "检测到欠拟合，减少dropout率"
            else:
                optimized_value = current_value
                improvement_metric = "stability"
                improvement_value = 0.02
                adjustment_reason = "保持当前dropout率"
                
        else:
            optimized_value = current_value
            improvement_metric = "stability"
            improvement_value = 0.05
            adjustment_reason = f"保持{parameter_name}当前配置以确保稳定性"
            
        return optimized_value, improvement_metric, improvement_value, adjustment_reason

    def _rule_based_knowledge_inference(self, query: str) -> Dict[str, Any]:
        """基于规则的知识推理"""
        entities = []
        relationships = []
        recommendations = []
        
        query_lower = query.lower()
        
        if any(k in query_lower for k in ['dataset', 'data', '数据集', '数据']):
            entities.append({"id": "dataset_entity", "type": "dataset", "name": "推断的数据集实体"})
            recommendations.append("建议进行数据预处理和特征工程")
            
        if any(k in query_lower for k in ['model', 'architecture', '模型', '架构']):
            entities.append({"id": "model_entity", "type": "model", "name": "推断的模型实体"})
            if any(k in query_lower for k in ['nlp', 'text', '文本', '自然语言']):
                recommendations.append("推荐使用Transformer架构处理文本数据")
            elif any(k in query_lower for k in ['image', 'vision', '图像', '视觉']):
                recommendations.append("推荐使用CNN架构处理图像数据")
            else:
                recommendations.append("推荐根据数据类型选择合适的模型架构")
                
        if any(k in query_lower for k in ['training', 'hyperparameter', '训练', '超参数']):
            entities.append({"id": "training_entity", "type": "training", "name": "推断的训练实体"})
            recommendations.append("建议使用自适应学习率和早停策略")
            
        if not entities:
            entities.append({"id": "general_entity", "type": "general", "name": "通用AI任务实体"})
            recommendations.append("建议根据具体任务需求选择合适的方法")
            
        return {
            "query": query,
            "entities": entities,
            "relationships": relationships,
            "recommendations": recommendations,
            "inference_type": "rule_based"
        }

    # ==========================================================================
    # 辅助方法
    # ==========================================================================

    def _decision_to_dict(self, decision) -> Dict[str, Any]:
        """将决策转换为字典"""
        if decision is None:
            return None
        if isinstance(decision, dict):
            return decision
        if hasattr(decision, 'to_dict'):
            return decision.to_dict()
        return asdict(decision)

    def _adaptive_opt_to_dict(self, opt) -> Dict[str, Any]:
        """将自适应优化转换为字典"""
        if opt is None:
            return None
        if isinstance(opt, dict):
            return opt
        if hasattr(opt, 'to_dict'):
            return opt.to_dict()
        return asdict(opt)

    def _knowledge_to_dict(self, knowledge) -> Dict[str, Any]:
        """将知识转换为字典"""
        if knowledge is None:
            return None
        if isinstance(knowledge, dict):
            return knowledge
        if hasattr(knowledge, 'to_dict'):
            return knowledge.to_dict()
        return asdict(knowledge)

    def _experience_to_dict(self, experience) -> Dict[str, Any]:
        """将经验转换为字典"""
        if experience is None:
            return None
        if isinstance(experience, dict):
            return experience
        if hasattr(experience, 'to_dict'):
            return experience.to_dict()
        return asdict(experience)


# 全局智能决策服务实例
_global_intelligent_decision_service = None


def get_intelligent_decision_service(use_memory_storage: bool = True) -> IntelligentDecisionService:
    """获取全局智能决策服务实例
    
    Args:
        use_memory_storage: 是否使用内存存储
    
    Returns:
        IntelligentDecisionService: 智能决策服务实例
    """
    global _global_intelligent_decision_service
    
    if _global_intelligent_decision_service is None:
        _global_intelligent_decision_service = IntelligentDecisionService(use_memory_storage)
        
    return _global_intelligent_decision_service

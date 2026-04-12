"""资源优化模块

提供智能的资源监控、性能分析和优化建议功能。
包含以下子模块：
- memory: 内存优化
- fusion: 算子融合优化
- folding: 常量折叠优化
- elimination: 冗余消除优化
- pruning: 模型剪枝优化
- graph: 图优化
"""

import logging

logger = logging.getLogger(__name__)

# ==================== 核心监控服务 ====================
try:
    from backend.core.monitoring.service import (
        UnifiedMonitoringService,
        get_monitoring_service as get_core_monitoring_service
    )
    from backend.core.monitoring.analyzer import (
        PerformanceAnalyzer,
        get_performance_analyzer as get_core_analyzer
    )
    from backend.core.monitoring.optimizer import (
        ResourceOptimizer,
        get_resource_optimizer as get_core_optimizer
    )
    from backend.core.monitoring.models import (
        SystemMetrics, GPUMetrics, TrainingMetrics,
        AlertRule, Alert, MetricType, AlertLevel
    )
    from backend.core.monitoring.exceptions import (
        MonitoringError, MetricsCollectionError, 
        AlertProcessingError, ResourceUnavailableError
    )
    
    # 为了向后兼容，重新导出核心服务的类和函数
    ResourceMonitor = UnifiedMonitoringService
    PerformanceAnalysisEngine = PerformanceAnalyzer
    ResourceOptimizerEngine = ResourceOptimizer
    
    get_resource_monitor = get_core_monitoring_service
    get_performance_analyzer = get_core_analyzer
    get_resource_optimizer = get_core_optimizer
    
    HAS_CORE_MONITORING = True
except ImportError as e:
    logger.warning(f"Core monitoring not available: {e}")
    HAS_CORE_MONITORING = False
    
    # 占位类
    class ResourceMonitor:
        pass
    class PerformanceAnalysisEngine:
        pass
    class ResourceOptimizerEngine:
        pass
    
    def get_resource_monitor():
        raise ImportError("Core monitoring service not available")
    
    def get_performance_analyzer():
        raise ImportError("Performance analyzer not available")
    
    def get_resource_optimizer():
        raise ImportError("Resource optimizer not available")

# ==================== 配置管理 ====================
from .models import (
    OptimizationConfig,
    OptimizationMode,
    OptimizationStrategy,
    ResourceType,
    ThresholdConfig,
    SchedulingConfig,
    CacheConfig,
    PredictionConfig,
)
from .manager import (
    ConfigManager,
    get_optimization_config,
    set_optimization_config,
    load_config_from_file,
    save_config_to_file,
)
from .optimization_errors import (
    OptimizationError,
    ResourceMonitorError,
    ResourceOptimizerError,
    PerformanceAnalyzerError,
    PredictionError,
    ConfigurationError,
    ResourceUnavailableError as OptResourceUnavailableError,
    InvalidRecommendationError,
)

# ==================== 内存优化 ====================
try:
    from .memory import (
        MemoryOptimizer,
        MemoryAnalyzer,
        MemoryTransformer,
        MemoryOptimizationLevel,
        MemoryProfile,
        MemoryOptimizationResult,
    )
    HAS_MEMORY_MODULE = True
except ImportError as e:
    logger.warning(f"Memory optimization module not available: {e}")
    HAS_MEMORY_MODULE = False

# ==================== 算子融合 ====================
try:
    from .fusion import (
        FusionOptimizer,
        FusionAnalyzer,
        FusionPatterns,
        FusionType,
        FusionResult,
    )
    HAS_FUSION_MODULE = True
except ImportError as e:
    logger.warning(f"Fusion optimization module not available: {e}")
    HAS_FUSION_MODULE = False

# ==================== 常量折叠 ====================
try:
    from .folding import (
        FoldingOptimizer,
        ConstantAnalyzer,
        FoldingTransformer,
        ConstantType,
        FoldingResult,
    )
    HAS_FOLDING_MODULE = True
except ImportError as e:
    logger.warning(f"Folding optimization module not available: {e}")
    HAS_FOLDING_MODULE = False

# ==================== 冗余消除 ====================
try:
    from .elimination import (
        EliminationOptimizer,
        DeadCodeAnalyzer,
        EliminationTransformer,
        RedundancyType,
        EliminationResult,
    )
    HAS_ELIMINATION_MODULE = True
except ImportError as e:
    logger.warning(f"Elimination optimization module not available: {e}")
    HAS_ELIMINATION_MODULE = False

# ==================== 模型剪枝 ====================
try:
    from .pruning import (
        PruningManager,
        PruningAnalysis,
        StructuredPruning,
        UnstructuredPruning,
        ChannelPruning,
        LayerPruning,
        GradualPruning,
        PruningResult,
    )
    HAS_PRUNING_MODULE = True
except ImportError as e:
    logger.warning(f"Pruning optimization module not available: {e}")
    HAS_PRUNING_MODULE = False

# ==================== 图优化 ====================
try:
    from .graph import (
        GraphOptimizer,
    )
    HAS_GRAPH_MODULE = True
except ImportError as e:
    logger.warning(f"Graph optimization module not available: {e}")
    HAS_GRAPH_MODULE = False

# ==================== 资源集成 ====================
try:
    from .resource_integration import (
        ResourceIntegrationManager,
        get_resource_integration_manager,
        create_resource_integration_manager,
        DEFAULT_INTEGRATION_CONFIG,
    )
    HAS_RESOURCE_INTEGRATION = True
except ImportError as e:
    logger.warning(f"Resource integration not available: {e}")
    HAS_RESOURCE_INTEGRATION = False

# ==================== 导出 ====================

__all__ = [
    # 配置模型
    'OptimizationConfig',
    'OptimizationMode',
    'OptimizationStrategy',
    'ResourceType',
    'ThresholdConfig',
    'SchedulingConfig',
    'CacheConfig',
    'PredictionConfig',

    # 配置管理
    'ConfigManager',
    'get_optimization_config',
    'set_optimization_config',
    'load_config_from_file',
    'save_config_to_file',

    # 异常
    'OptimizationError',
    'ResourceMonitorError',
    'ResourceOptimizerError',
    'PerformanceAnalyzerError',
    'PredictionError',
    'ConfigurationError',
    'InvalidRecommendationError',

    # 资源监控 (使用统一服务)
    'ResourceMonitor',
    'get_resource_monitor',

    # 资源优化 (使用统一服务)
    'ResourceOptimizerEngine',
    'get_resource_optimizer',

    # 性能分析 (使用统一服务)
    'PerformanceAnalysisEngine',
    'get_performance_analyzer',
    
    # 内存优化
    'MemoryOptimizer',
    'MemoryAnalyzer',
    'MemoryTransformer',
    'MemoryOptimizationLevel',
    'MemoryProfile',
    'MemoryOptimizationResult',
    
    # 算子融合
    'FusionOptimizer',
    'FusionAnalyzer',
    'FusionPatterns',
    'FusionType',
    'FusionResult',
    
    # 常量折叠
    'FoldingOptimizer',
    'ConstantAnalyzer',
    'FoldingTransformer',
    'ConstantType',
    'FoldingResult',
    
    # 冗余消除
    'EliminationOptimizer',
    'DeadCodeAnalyzer',
    'EliminationTransformer',
    'RedundancyType',
    'EliminationResult',
    
    # 模型剪枝
    'PruningManager',
    'PruningAnalysis',
    'StructuredPruning',
    'UnstructuredPruning',
    'ChannelPruning',
    'LayerPruning',
    'GradualPruning',
    'PruningResult',
    
    # 图优化
    'GraphOptimizer',

    # 资源集成
    'ResourceIntegrationManager',
    'get_resource_integration_manager',
    'create_resource_integration_manager',
    'DEFAULT_INTEGRATION_CONFIG',
    
    # 模块可用性标志
    'HAS_CORE_MONITORING',
    'HAS_MEMORY_MODULE',
    'HAS_FUSION_MODULE',
    'HAS_FOLDING_MODULE',
    'HAS_ELIMINATION_MODULE',
    'HAS_PRUNING_MODULE',
    'HAS_GRAPH_MODULE',
    'HAS_RESOURCE_INTEGRATION',
]

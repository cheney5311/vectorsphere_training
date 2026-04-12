"""性能分析引擎 - 已重构为使用统一核心服务

此文件已重构为使用统一的性能分析核心服务，避免重复代码。
"""

from backend.core.monitoring.analyzer import (
    PerformanceAnalyzer,
    get_performance_analyzer as get_core_performance_analyzer
)

# 为了向后兼容，重新导出核心服务的类和函数
PerformanceAnalysisEngine = PerformanceAnalyzer
get_performance_analyzer = get_core_performance_analyzer

__all__ = [
    'PerformanceAnalysisEngine',
    'get_performance_analyzer'
]

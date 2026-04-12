"""资源优化器 - 已重构为使用统一核心服务

此文件已重构为使用统一的资源优化核心服务，避免重复代码。
"""

from backend.core.monitoring.optimizer import (
    ResourceOptimizer,
    get_resource_optimizer as get_core_resource_optimizer
)

# 为了向后兼容，重新导出核心服务的类和函数
ResourceOptimizerEngine = ResourceOptimizer
get_resource_optimizer = get_core_resource_optimizer

# 创建一个别名以保持向后兼容
create_resource_optimizer = lambda config=None: ResourceOptimizer(config)

__all__ = [
    'ResourceOptimizerEngine',
    'get_resource_optimizer',
    'create_resource_optimizer'
]

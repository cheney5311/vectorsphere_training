"""训练插件模块

提供生产级训练插件系统：
- TrainingPlugin: 插件基类
- CallbackPlugin: 回调插件
- MonitoringPlugin: 监控插件
- CheckpointPlugin: 检查点插件
- PluginRegistry: 插件注册表

架构位置：
┌─────────────────────────────────────────────────────────────┐
│  orchestrator / pipeline                                     │
│    └── >>> plugins (本模块) <<<                              │
│        ├── TrainingPlugin (插件基类)                         │
│        ├── PluginRegistry (插件注册表)                       │
│        └── HookPoint (钩子点)                                │
│            └── strategies/* (策略层)                         │
└─────────────────────────────────────────────────────────────┘

使用示例：
```python
from backend.modules.training.plugins import (
    CallbackPlugin, PluginConfig, PluginContext, PluginResult,
    HookPoint, register_plugin, execute_hook,
)

class MyPlugin(CallbackPlugin):
    def on_epoch_end(self, context):
        print(f"Epoch {context.epoch} ended with loss {context.metrics.get('loss')}")
        return PluginResult()

# 注册插件
plugin = MyPlugin()
register_plugin(plugin)

# 执行钩子
context = PluginContext(
    hook=HookPoint.ON_EPOCH_END,
    epoch=1,
    metrics={'loss': 0.5}
)
results = execute_hook(HookPoint.ON_EPOCH_END, context)
```
"""

# 接口
from .interface import (
    # 基类
    TrainingPlugin,
    CallbackPlugin,
    MonitoringPlugin,
    CheckpointPlugin,
    
    # 数据类
    PluginConfig,
    PluginContext,
    PluginResult,
    
    # 枚举
    PluginType,
    PluginPriority,
    HookPoint,
    
    # 便捷函数
    create_plugin_config,
    create_plugin_context,
    
    # 层可用性
    STRATEGY_LAYER_AVAILABLE,
)

# 别名
PluginInterface = TrainingPlugin

# 注册表
from .registry import (
    # 注册表
    PluginRegistry,
    registry,
    get_plugin_registry,
    
    # 装饰器
    plugin,
    
    # 便捷函数
    register_plugin,
    unregister_plugin,
    get_plugin,
    execute_hook,
    create_plugin,
)


# ==================== 便捷函数 ====================

def diagnose_plugin_module() -> dict:
    """诊断插件模块"""
    return {
        'module': 'plugins',
        'plugin_count': len(registry.get_all()),
        'hook_points': list(HookPoint.__members__.keys()),
        'plugin_types': list(PluginType.__members__.keys()),
        'classes': {
            'TrainingPlugin': TrainingPlugin is not None,
            'CallbackPlugin': CallbackPlugin is not None,
            'MonitoringPlugin': MonitoringPlugin is not None,
            'CheckpointPlugin': CheckpointPlugin is not None,
            'PluginRegistry': PluginRegistry is not None,
        },
        'strategy_layer_available': STRATEGY_LAYER_AVAILABLE,
    }


__all__ = [
    # 基类
    'TrainingPlugin',
    'PluginInterface',  # 别名
    'CallbackPlugin',
    'MonitoringPlugin',
    'CheckpointPlugin',
    
    # 数据类
    'PluginConfig',
    'PluginContext',
    'PluginResult',
    
    # 枚举
    'PluginType',
    'PluginPriority',
    'HookPoint',
    
    # 注册表
    'PluginRegistry',
    'registry',
    'get_plugin_registry',
    
    # 装饰器
    'plugin',
    
    # 便捷函数
    'create_plugin_config',
    'create_plugin_context',
    'register_plugin',
    'unregister_plugin',
    'get_plugin',
    'execute_hook',
    'create_plugin',
    'diagnose_plugin_module',
    
    # 层可用性
    'STRATEGY_LAYER_AVAILABLE',
]

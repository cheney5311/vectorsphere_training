"""训练插件注册表

生产级插件注册和管理系统，支持：
- 插件注册和发现
- 插件生命周期管理
- 钩子执行调度
- 策略层集成

架构位置：
├── plugins/registry.py (本模块)
│   └── 插件注册和管理
├── plugins/interface.py
│   └── 插件接口定义
└── 被 orchestrator, pipeline 调用
"""

import logging
import threading
from typing import Dict, Any, Optional, List, Type, Callable, Union
from datetime import datetime
from collections import defaultdict

from .interface import (
    TrainingPlugin, CallbackPlugin, MonitoringPlugin, CheckpointPlugin,
    PluginConfig, PluginContext, PluginResult,
    PluginType, PluginPriority, HookPoint,
    STRATEGY_LAYER_AVAILABLE,
)

logger = logging.getLogger(__name__)


# ==================== 策略层导入 ====================

StrategyContext = None

if STRATEGY_LAYER_AVAILABLE:
    try:
        from backend.modules.training.strategies.base_strategy import StrategyContext
    except (ImportError, SyntaxError):
        pass


# ==================== 插件注册表 ====================

class PluginRegistry:
    """插件注册表
    
    管理训练插件的注册、发现和执行。
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, *args, **kwargs):
        """单例模式"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if getattr(self, '_initialized', False):
            return
        
        # 插件存储
        self._plugins: Dict[str, TrainingPlugin] = {}
        self._plugin_classes: Dict[str, Type[TrainingPlugin]] = {}
        
        # 钩子到插件的映射
        self._hook_plugins: Dict[HookPoint, List[str]] = defaultdict(list)
        
        # 类型到插件的映射
        self._type_plugins: Dict[PluginType, List[str]] = defaultdict(list)
        
        # 锁
        self._registry_lock = threading.RLock()
        
        # 注册内置插件
        self._register_builtin_plugins()
        
        self._initialized = True
        logger.info("PluginRegistry initialized")
    
    def _register_builtin_plugins(self) -> None:
        """注册内置插件"""
        # 注册内置插件类
        self.register_class('callback', CallbackPlugin)
        self.register_class('monitoring', MonitoringPlugin)
        self.register_class('checkpoint', CheckpointPlugin)
    
    def register_class(self, name: str, plugin_class: Type[TrainingPlugin]) -> None:
        """注册插件类
        
        Args:
            name: 插件类名称
            plugin_class: 插件类
        """
        with self._registry_lock:
            self._plugin_classes[name] = plugin_class
            logger.debug(f"Plugin class registered: {name}")
    
    def register(
        self, 
        plugin: TrainingPlugin,
        hooks: Optional[List[HookPoint]] = None
    ) -> bool:
        """注册插件实例
        
        Args:
            plugin: 插件实例
            hooks: 要注册的钩子列表
        
        Returns:
            是否注册成功
        """
        with self._registry_lock:
            name = plugin.name
            
            if name in self._plugins:
                logger.warning(f"Plugin already registered: {name}")
                return False
            
            # 初始化插件
            if not plugin.initialize():
                logger.error(f"Plugin initialization failed: {name}")
                return False
            
            # 存储插件
            self._plugins[name] = plugin
            
            # 注册钩子
            effective_hooks = hooks or plugin.config.hooks or list(HookPoint)
            for hook in effective_hooks:
                self._hook_plugins[hook].append(name)
            
            # 注册类型
            self._type_plugins[plugin.plugin_type].append(name)
            
            # 按优先级排序钩子列表
            for hook in effective_hooks:
                self._hook_plugins[hook] = sorted(
                    self._hook_plugins[hook],
                    key=lambda n: self._plugins[n].priority if n in self._plugins else 100
                )
            
            logger.info(f"Plugin registered: {name} (type={plugin.plugin_type.value})")
            return True
    
    def unregister(self, name: str) -> bool:
        """取消注册插件
        
        Args:
            name: 插件名称
        
        Returns:
            是否取消成功
        """
        with self._registry_lock:
            if name not in self._plugins:
                return False
            
            plugin = self._plugins[name]
            
            # 关闭插件
            plugin.shutdown()
            
            # 从钩子映射中移除
            for hook in HookPoint:
                if name in self._hook_plugins[hook]:
                    self._hook_plugins[hook].remove(name)
            
            # 从类型映射中移除
            if name in self._type_plugins[plugin.plugin_type]:
                self._type_plugins[plugin.plugin_type].remove(name)
            
            # 删除插件
            del self._plugins[name]
            
            logger.info(f"Plugin unregistered: {name}")
            return True
    
    def get(self, name: str) -> Optional[TrainingPlugin]:
        """获取插件
        
        Args:
            name: 插件名称
        
        Returns:
            插件实例或 None
        """
        return self._plugins.get(name)
    
    def get_all(self) -> List[TrainingPlugin]:
        """获取所有插件"""
        return list(self._plugins.values())
    
    def get_by_type(self, plugin_type: PluginType) -> List[TrainingPlugin]:
        """按类型获取插件
        
        Args:
            plugin_type: 插件类型
        
        Returns:
            插件列表
        """
        names = self._type_plugins.get(plugin_type, [])
        return [self._plugins[n] for n in names if n in self._plugins]
    
    def get_by_hook(self, hook: HookPoint) -> List[TrainingPlugin]:
        """按钩子获取插件
        
        Args:
            hook: 钩子点
        
        Returns:
            插件列表（按优先级排序）
        """
        names = self._hook_plugins.get(hook, [])
        return [self._plugins[n] for n in names if n in self._plugins]
    
    def execute_hook(
        self, 
        hook: HookPoint, 
        context: PluginContext
    ) -> List[PluginResult]:
        """执行钩子
        
        Args:
            hook: 钩子点
            context: 插件上下文
        
        Returns:
            执行结果列表
        """
        results = []
        context.hook = hook
        
        plugins = self.get_by_hook(hook)
        
        for plugin in plugins:
            if not plugin.enabled:
                continue
            
            if not plugin.supports_hook(hook):
                continue
            
            try:
                result = plugin.execute(context)
                results.append(result)
                
                # 检查是否应该停止
                if result.should_stop:
                    logger.info(f"Plugin {plugin.name} requested stop")
                    break
                
                # 更新上下文数据
                if result.modified_data:
                    context.data.update(result.modified_data)
                    
            except Exception as e:
                logger.error(f"Plugin {plugin.name} execution failed: {e}")
                results.append(PluginResult(
                    success=False,
                    message=str(e),
                ))
        
        return results
    
    def create_plugin(
        self, 
        class_name: str, 
        config: Optional[PluginConfig] = None,
        **kwargs
    ) -> Optional[TrainingPlugin]:
        """创建插件实例
        
        Args:
            class_name: 插件类名称
            config: 插件配置
            **kwargs: 额外参数
        
        Returns:
            插件实例或 None
        """
        if class_name not in self._plugin_classes:
            logger.error(f"Unknown plugin class: {class_name}")
            return None
        
        plugin_class = self._plugin_classes[class_name]
        
        try:
            if config:
                return plugin_class(config)
            return plugin_class()
        except Exception as e:
            logger.error(f"Failed to create plugin {class_name}: {e}")
            return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取注册表统计信息"""
        with self._registry_lock:
            return {
                'total_plugins': len(self._plugins),
                'registered_classes': list(self._plugin_classes.keys()),
                'plugins_by_type': {
                    t.value: len(plugins) 
                    for t, plugins in self._type_plugins.items()
                    if plugins
                },
                'hooks_with_plugins': {
                    h.value: len(plugins)
                    for h, plugins in self._hook_plugins.items()
                    if plugins
                },
                'plugins': [p.get_info() for p in self._plugins.values()],
            }
    
    def clear(self) -> None:
        """清空注册表"""
        with self._registry_lock:
            for plugin in list(self._plugins.values()):
                plugin.shutdown()
            
            self._plugins.clear()
            self._hook_plugins.clear()
            self._type_plugins.clear()
            
            logger.info("PluginRegistry cleared")


# ==================== 全局实例 ====================

registry = PluginRegistry()


def get_plugin_registry() -> PluginRegistry:
    """获取全局插件注册表"""
    return registry


# ==================== 装饰器 ====================

def plugin(
    name: Optional[str] = None,
    plugin_type: PluginType = PluginType.CALLBACK,
    priority: PluginPriority = PluginPriority.NORMAL,
    hooks: Optional[List[HookPoint]] = None,
    auto_register: bool = True
):
    """插件装饰器
    
    用于将类标记为插件并自动注册。
    
    Args:
        name: 插件名称
        plugin_type: 插件类型
        priority: 优先级
        hooks: 支持的钩子
        auto_register: 是否自动注册类
    
    Example:
        @plugin(name="my_plugin", plugin_type=PluginType.CALLBACK)
        class MyPlugin(CallbackPlugin):
            def on_epoch_end(self, context):
                print(f"Epoch {context.epoch} ended")
                return PluginResult()
    """
    def decorator(cls: Type[TrainingPlugin]):
        # 设置默认配置
        original_init = cls.__init__
        
        def new_init(self, config=None, **kwargs):
            if config is None:
                config = PluginConfig(
                    name=name or cls.__name__,
                    plugin_type=plugin_type,
                    priority=priority,
                    hooks=hooks or [],
                )
            original_init(self, config, **kwargs)
        
        cls.__init__ = new_init
        
        # 自动注册类
        if auto_register:
            registry.register_class(name or cls.__name__, cls)
        
        return cls
    
    return decorator


# ==================== 便捷函数 ====================

def register_plugin(plugin: TrainingPlugin, hooks: Optional[List[HookPoint]] = None) -> bool:
    """注册插件"""
    return registry.register(plugin, hooks)


def unregister_plugin(name: str) -> bool:
    """取消注册插件"""
    return registry.unregister(name)


def get_plugin(name: str) -> Optional[TrainingPlugin]:
    """获取插件"""
    return registry.get(name)


def execute_hook(hook: HookPoint, context: PluginContext) -> List[PluginResult]:
    """执行钩子"""
    return registry.execute_hook(hook, context)


def create_plugin(class_name: str, **kwargs) -> Optional[TrainingPlugin]:
    """创建插件"""
    return registry.create_plugin(class_name, **kwargs)


# ==================== 导出 ====================

__all__ = [
    # 注册表
    'PluginRegistry',
    'registry',
    'get_plugin_registry',
    
    # 装饰器
    'plugin',
    
    # 便捷函数
    'register_plugin',
    'unregister_plugin',
    'get_plugin',
    'execute_hook',
    'create_plugin',
]

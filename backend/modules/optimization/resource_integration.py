"""资源优化集成管理器

将资源优化算法与资源监控系统和任务调度系统集成，实现智能资源管理。
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
from threading import RLock

from backend.services.resource_monitor import ResourceMonitor
from backend.services.resource_optimizer import ResourceOptimizer
from backend.core.monitoring.models import MetricPoint, Alert
from backend.modules.optimization.optimization_errors import ResourceUnavailableError

logger = logging.getLogger(__name__)


class ResourceIntegrationManager:
    """资源集成管理器

    将资源监控、资源优化和任务调度系统集成，实现智能资源管理。
    """

    def __init__(
        self,
        resource_monitor: ResourceMonitor,
        resource_optimizer: ResourceOptimizer,
        config: Dict[str, Any] = {}
    ):
        self.resource_monitor = resource_monitor
        self.resource_optimizer = resource_optimizer
        self.config = config or DEFAULT_INTEGRATION_CONFIG

        # 状态
        self._running = False
        self._integration_task: Optional[asyncio.Task] = None
        self._lock = RLock()

        # 优化历史
        self.optimization_history: List[Dict[str, Any]] = []

        # 回调函数
        self.on_optimization_applied: Optional[Callable[[Dict[str, Any]], None]] = None
        self.on_resource_alert: Optional[Callable[[Alert], None]] = None

        # 注册告警回调
        self.resource_monitor.alert_manager.add_alert_callback(self._on_resource_alert)

    async def start(self):
        """启动集成管理器"""
        if self._running:
            return

        self._running = True
        logger.info("启动资源集成管理器")

        # 启动资源监控
        await self.resource_monitor.start()

        # 启动资源优化器
        # Note: ResourceOptimizer may not have start method, check if available
        if hasattr(self.resource_optimizer, 'start'):
            await self.resource_optimizer.start()

        # 启动集成循环
        self._integration_task = asyncio.create_task(self._integration_loop())

    async def stop(self):
        """停止集成管理器"""
        if not self._running:
            return

        self._running = False
        logger.info("停止资源集成管理器")

        # 停止资源优化器
        # Note: ResourceOptimizer may not have stop method, check if available
        if hasattr(self.resource_optimizer, 'stop'):
            await self.resource_optimizer.stop()

        # 停止资源监控
        await self.resource_monitor.stop()

        # 取消集成任务
        if self._integration_task:
            self._integration_task.cancel()
            try:
                await self._integration_task
            except asyncio.CancelledError:
                pass

    async def _integration_loop(self):
        """集成循环"""
        integration_interval = self.config.get('integration_interval_seconds', 60)

        while self._running:
            try:
                # 执行集成逻辑
                await self._perform_integration()

                # 等待下一次集成
                await asyncio.sleep(integration_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"集成循环错误: {e}")
                await asyncio.sleep(10)  # 10秒后重试

    async def _perform_integration(self):
        """执行集成逻辑"""
        with self._lock:
            # 1. 获取当前资源状态
            current_metrics = self.resource_monitor.get_current_metrics()
            system_overview = self.resource_monitor.get_system_overview()

            # 2. 根据资源状态，生成优化建议
            recommendations = await self._generate_integrated_recommendations(
                current_metrics, system_overview
            )

            # 3. 应用优化建议
            if recommendations:
                applied_recommendations = await self._apply_integrated_recommendations(recommendations)

                # 记录优化历史
                if applied_recommendations:
                    optimization_record = {
                        'timestamp': datetime.now(),
                        'system_status': system_overview['status'],
                        'recommendations_count': len(recommendations),
                        'applied_count': len(applied_recommendations),
                        'recommendations': applied_recommendations
                    }

                    self.optimization_history.append(optimization_record)

                    # 触发回调
                    if callable(self.on_optimization_applied):
                        self.on_optimization_applied(optimization_record)

    async def _generate_integrated_recommendations(
        self,
        current_metrics: Dict[str, MetricPoint],
        system_overview: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """生成集成优化建议"""
        recommendations = []

        # 1. 基于资源监控的建议
        if system_overview['status'] != 'healthy':
            # 系统状态不健康，生成资源调整建议
            if 'cpu_utilization' in system_overview and system_overview['cpu_utilization'] > 0.9:
                recommendations.append({
                    'type': 'resource_adjustment',
                    'resource': 'cpu',
                    'action': 'reduce_load',
                    'priority': 'high',
                    'description': 'CPU利用率过高，建议减少任务负载或增加资源'
                })

            if 'memory_utilization' in system_overview and system_overview['memory_utilization'] > 0.85:
                recommendations.append({
                    'type': 'resource_adjustment',
                    'resource': 'memory',
                    'action': 'free_memory',
                    'priority': 'high',
                    'description': '内存利用率过高，建议释放缓存或增加资源'
                })

            if 'gpu_utilization' in system_overview:
                for gpu_id, utilization in system_overview['gpu_utilization'].items():
                    if utilization > 0.95:
                        recommendations.append({
                            'type': 'resource_adjustment',
                            'resource': 'gpu',
                            'gpu_id': gpu_id.split('_')[1],
                            'action': 'balance_load',
                            'priority': 'high',
                            'description': f'GPU {gpu_id} 利用率过高，建议平衡负载'
                        })

        return recommendations

    async def _apply_integrated_recommendations(
        self, recommendations: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """应用集成优化建议"""
        applied = []
        for recommendation in recommendations:
            try:
                # 这里可以实现具体的优化建议应用逻辑
                # 为简化起见，我们只记录建议
                logger.info(f"应用集成优化建议: {recommendation['description']}")
                applied.append(recommendation)
            except Exception as e:
                logger.error(f"应用集成优化建议失败 {recommendation['description']}: {e}")

        return applied

    def _on_resource_alert(self, alert: Alert):
        """处理资源告警

        Args:
            alert: 告警信息
        """
        logger.warning(f"收到资源告警: {alert.title} - {alert.message}")

        # 触发回调
        if callable(self.on_resource_alert):
            try:
                self.on_resource_alert(alert)
            except Exception as e:
                logger.error(f"资源告警回调执行失败: {e}")


# 全局资源集成管理器实例
_global_integration_manager: Optional[ResourceIntegrationManager] = None


def get_resource_integration_manager() -> ResourceIntegrationManager:
    """获取全局资源集成管理器实例

    Returns:
        ResourceIntegrationManager: 资源集成管理器实例
    """
    global _global_integration_manager
    if _global_integration_manager is None:
        # 获取依赖组件
        resource_monitor = get_resource_monitor()
        resource_optimizer = get_resource_optimizer()

        _global_integration_manager = ResourceIntegrationManager(
            resource_monitor, resource_optimizer
        )
    return _global_integration_manager


def create_resource_integration_manager(
    resource_monitor: ResourceMonitor,
    resource_optimizer: ResourceOptimizer,
    config: Optional[Dict[str, Any]] = None
) -> ResourceIntegrationManager:
    """创建资源集成管理器实例

    Args:
        resource_monitor: 资源监控器
        resource_optimizer: 资源优化器
        config: 配置参数

    Returns:
        ResourceIntegrationManager: 资源集成管理器实例
    """
    return ResourceIntegrationManager(resource_monitor, resource_optimizer, config)


# 默认配置
DEFAULT_INTEGRATION_CONFIG = {
    'integration_interval_seconds': 60,
    'alert_processing_enabled': True,
    'auto_optimization_enabled': True
}


# 导入依赖组件的函数
def get_resource_monitor():
    """获取资源监控器实例"""
    from backend.services.resource_monitor import get_resource_monitor as _get_resource_monitor
    return _get_resource_monitor()


def get_resource_optimizer():
    """获取资源优化器实例"""
    from backend.services.resource_optimizer import get_resource_optimizer as _get_resource_optimizer
    return _get_resource_optimizer()
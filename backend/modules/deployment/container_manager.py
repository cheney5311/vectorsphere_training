"""容器管理器

提供容器化部署的管理功能，包括：
- 容器部署和销毁
- 扩缩容管理
- 发布策略执行（滚动更新、金丝雀、蓝绿、AB测试）
- 健康检查和状态监控
- 回滚操作
"""

import logging
import uuid
import time
import threading
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from enum import Enum

from .deployment_config import DeploymentConfig
from .deployment_exceptions import ContainerDeploymentError, DeploymentNotFoundError

logger = logging.getLogger(__name__)


class ContainerStatus(Enum):
    """容器状态枚举"""
    PENDING = "pending"
    CREATING = "creating"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"
    TERMINATED = "terminated"


class ReleasePhase(Enum):
    """发布阶段枚举"""
    IDLE = "idle"
    ROLLING = "rolling"
    CANARY = "canary"
    BLUE_GREEN = "blue_green"
    AB_TESTING = "ab_testing"


@dataclass
class ContainerInstance:
    """容器实例信息"""
    instance_id: str
    deployment_id: str
    status: ContainerStatus = ContainerStatus.PENDING
    port: int = 8080
    health_status: str = "unknown"
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    request_count: int = 0
    error_count: int = 0
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    version: str = "v1"
    traffic_weight: float = 100.0  # 流量权重百分比
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'instance_id': self.instance_id,
            'deployment_id': self.deployment_id,
            'status': self.status.value,
            'port': self.port,
            'health_status': self.health_status,
            'cpu_usage': self.cpu_usage,
            'memory_usage': self.memory_usage,
            'request_count': self.request_count,
            'error_count': self.error_count,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'version': self.version,
            'traffic_weight': self.traffic_weight
        }


@dataclass
class DeploymentState:
    """部署状态信息"""
    deployment_id: str
    model_id: str
    status: ContainerStatus = ContainerStatus.PENDING
    desired_replicas: int = 1
    available_replicas: int = 0
    ready_replicas: int = 0
    instances: List[ContainerInstance] = field(default_factory=list)
    endpoint_url: str = ""
    health_check_url: str = ""
    release_phase: ReleasePhase = ReleasePhase.IDLE
    current_version: str = "v1"
    previous_version: Optional[str] = None
    canary_percent: int = 0
    ab_test_percent: int = 50
    blue_active: bool = True  # True=蓝色环境活跃，False=绿色环境活跃
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    config: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'deployment_id': self.deployment_id,
            'model_id': self.model_id,
            'status': self.status.value,
            'desired_replicas': self.desired_replicas,
            'available_replicas': self.available_replicas,
            'ready_replicas': self.ready_replicas,
            'instances': [inst.to_dict() for inst in self.instances],
            'endpoint_url': self.endpoint_url,
            'health_check_url': self.health_check_url,
            'release_phase': self.release_phase.value,
            'current_version': self.current_version,
            'previous_version': self.previous_version,
            'canary_percent': self.canary_percent,
            'ab_test_percent': self.ab_test_percent,
            'blue_active': self.blue_active,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }


class ContainerManager:
    """容器管理器
    
    负责容器的生命周期管理，包括部署、扩缩容、发布策略执行等。
    支持多种发布策略：滚动更新、金丝雀发布、蓝绿部署、AB测试。
    """
    
    def __init__(self):
        """初始化容器管理器"""
        self._deployments: Dict[str, DeploymentState] = {}
        self._lock = threading.RLock()
        self._base_port = 8080
        self._port_counter = 0
        logger.info("ContainerManager initialized")
    
    def _generate_instance_id(self) -> str:
        """生成容器实例ID"""
        return f"inst_{uuid.uuid4().hex[:12]}"
    
    def _allocate_port(self) -> int:
        """分配端口"""
        with self._lock:
            port = self._base_port + self._port_counter
            self._port_counter += 1
            return port
    
    def _create_instance(self, deployment_id: str, version: str = "v1", 
                        traffic_weight: float = 100.0) -> ContainerInstance:
        """创建容器实例
        
        Args:
            deployment_id: 部署ID
            version: 版本号
            traffic_weight: 流量权重
            
        Returns:
            容器实例
        """
        instance = ContainerInstance(
            instance_id=self._generate_instance_id(),
            deployment_id=deployment_id,
            status=ContainerStatus.CREATING,
            port=self._allocate_port(),
            version=version,
            traffic_weight=traffic_weight
        )
        
        # 模拟容器启动过程
        time.sleep(0.01)  # 模拟启动延迟
        instance.status = ContainerStatus.RUNNING
        instance.health_status = "healthy"
        instance.updated_at = datetime.utcnow()
        
        logger.info(f"Created container instance: {instance.instance_id} for deployment: {deployment_id}")
        return instance
    
    def _destroy_instance(self, instance: ContainerInstance) -> bool:
        """销毁容器实例
        
        Args:
            instance: 容器实例
            
        Returns:
            是否成功销毁
        """
        try:
            instance.status = ContainerStatus.STOPPING
            time.sleep(0.01)  # 模拟停止延迟
            instance.status = ContainerStatus.TERMINATED
            instance.health_status = "terminated"
            instance.updated_at = datetime.utcnow()
            logger.info(f"Destroyed container instance: {instance.instance_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to destroy instance {instance.instance_id}: {e}")
            return False
    
    def _check_instance_health(self, instance: ContainerInstance) -> bool:
        """检查实例健康状态
        
        Args:
            instance: 容器实例
            
        Returns:
            是否健康
        """
        if instance.status != ContainerStatus.RUNNING:
            return False
        
        # 模拟健康检查
        # 在实际实现中，这里应该发起 HTTP 请求到健康检查端点
        import random
        is_healthy = random.random() > 0.05  # 95% 概率健康
        
        instance.health_status = "healthy" if is_healthy else "unhealthy"
        instance.updated_at = datetime.utcnow()
        
        return is_healthy
    
    def _update_deployment_metrics(self, state: DeploymentState):
        """更新部署指标
        
        Args:
            state: 部署状态
        """
        running_instances = [i for i in state.instances if i.status == ContainerStatus.RUNNING]
        healthy_instances = [i for i in running_instances if i.health_status == "healthy"]
        
        state.available_replicas = len(running_instances)
        state.ready_replicas = len(healthy_instances)
        state.updated_at = datetime.utcnow()
        
        # 更新整体状态
        if state.available_replicas == 0:
            state.status = ContainerStatus.STOPPED
        elif state.ready_replicas < state.desired_replicas:
            state.status = ContainerStatus.CREATING
        else:
            state.status = ContainerStatus.RUNNING
    
    # ==================== 基础部署操作 ====================
        
    def deploy_container(self, deployment_id: str, config: DeploymentConfig) -> Dict[str, Any]:
        """部署容器
        
        Args:
            deployment_id: 部署ID
            config: 部署配置
            
        Returns:
            部署结果
            
        Raises:
            ContainerDeploymentError: 容器部署失败
        """
        try:
            logger.info(f"Starting container deployment: {deployment_id}, model: {config.model_id}")
            
            with self._lock:
                # 检查是否已存在
                if deployment_id in self._deployments:
                    raise ContainerDeploymentError(f"Deployment already exists: {deployment_id}", deployment_id)
                
                # 创建部署状态
                state = DeploymentState(
                    deployment_id=deployment_id,
                    model_id=config.model_id,
                    desired_replicas=config.replicas,
                    config=config.to_dict()
                )
                
                # 创建容器实例
                for _ in range(config.replicas):
                    instance = self._create_instance(deployment_id)
                    state.instances.append(instance)
                
                # 生成端点URL
                primary_port = state.instances[0].port if state.instances else config.port
                state.endpoint_url = f"http://localhost:{primary_port}/models/{config.model_id}/predict"
                state.health_check_url = f"http://localhost:{primary_port}{config.health_check_path}"
                state.status = ContainerStatus.RUNNING
                
                self._update_deployment_metrics(state)
                self._deployments[deployment_id] = state
                
                logger.info(f"Container deployment completed: {deployment_id}, replicas: {state.available_replicas}")
                
                return {
                'deployment_id': deployment_id,
                'model_id': config.model_id,
                    'status': state.status.value,
                    'replicas': state.available_replicas,
                    'endpoint_url': state.endpoint_url,
                    'port': primary_port,
                    'created_at': state.created_at.isoformat()
                }
                
        except ContainerDeploymentError:
            raise
        except Exception as e:
            logger.error(f"Container deployment failed: {e}")
            raise ContainerDeploymentError(f"Container deployment failed: {str(e)}", deployment_id)
    
    def deploy_service(self, deployment_id: str, container_config: Dict[str, Any]) -> str:
        """部署服务（简化接口）
        
        Args:
            deployment_id: 部署ID
            container_config: 容器配置字典
            
        Returns:
            端点URL
        """
        # 转换为 DeploymentConfig
        config = DeploymentConfig(
            model_id=container_config.get('image', '').replace('model-server:', ''),
            replicas=container_config.get('replicas', 1),
            environment_vars=container_config.get('environment', {}),
            custom_config=container_config
        )
        
        result = self.deploy_container(deployment_id, config)
        return result['endpoint_url']
    
    def get_container_status(self, deployment_id: str) -> Optional[Dict[str, Any]]:
        """获取容器状态
        
        Args:
            deployment_id: 部署ID
            
        Returns:
            容器状态信息
        """
        with self._lock:
            state = self._deployments.get(deployment_id)
            if not state:
                return None
            
            # 更新健康状态
            for instance in state.instances:
                if instance.status == ContainerStatus.RUNNING:
                    self._check_instance_health(instance)
            
            self._update_deployment_metrics(state)
            
            return state.to_dict()
    
    def get_deployment_status(self, deployment_id: str) -> Optional[Dict[str, Any]]:
        """获取部署状态（别名方法）"""
        return self.get_container_status(deployment_id)
    
    def stop_container(self, deployment_id: str) -> bool:
        """停止容器
        
        Args:
            deployment_id: 部署ID
            
        Returns:
            是否成功停止
        """
        try:
            with self._lock:
                state = self._deployments.get(deployment_id)
                if not state:
                    logger.warning(f"Deployment not found: {deployment_id}")
                    return False
                
                # 停止所有实例
                for instance in state.instances:
                    self._destroy_instance(instance)
                
                state.instances.clear()
                state.status = ContainerStatus.STOPPED
                self._update_deployment_metrics(state)
                
                del self._deployments[deployment_id]
                
                logger.info(f"Container stopped: {deployment_id}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to stop container: {e}")
            return False
    
    def stop_service(self, deployment_id: str) -> bool:
        """停止服务（别名方法）"""
        return self.stop_container(deployment_id)
    
    def list_containers(self) -> List[Dict[str, Any]]:
        """列出所有容器
        
        Returns:
            容器列表
        """
        with self._lock:
            return [state.to_dict() for state in self._deployments.values()]
    
    # ==================== 扩缩容操作 ====================
    
    def scale_up(self, deployment_id: str, target_replicas: int) -> bool:
        """扩容
        
        Args:
            deployment_id: 部署ID
            target_replicas: 目标副本数
            
        Returns:
            是否成功扩容
        """
        try:
            with self._lock:
                state = self._deployments.get(deployment_id)
                if not state:
                    raise DeploymentNotFoundError(f"Deployment not found: {deployment_id}", deployment_id)
                
                current_replicas = len([i for i in state.instances if i.status == ContainerStatus.RUNNING])
                
                if target_replicas <= current_replicas:
                    logger.info(f"No scale up needed: current={current_replicas}, target={target_replicas}")
                    return True
                
                # 创建新实例
                new_instances_count = target_replicas - current_replicas
                for _ in range(new_instances_count):
                    instance = self._create_instance(deployment_id, version=state.current_version)
                    state.instances.append(instance)
                
                state.desired_replicas = target_replicas
                self._update_deployment_metrics(state)
                
                logger.info(f"Scaled up deployment {deployment_id}: {current_replicas} -> {target_replicas}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to scale up: {e}")
            return False
    
    def scale_down(self, deployment_id: str, target_replicas: int) -> bool:
        """缩容
        
        Args:
            deployment_id: 部署ID
            target_replicas: 目标副本数
            
        Returns:
            是否成功缩容
        """
        try:
            with self._lock:
                state = self._deployments.get(deployment_id)
                if not state:
                    raise DeploymentNotFoundError(f"Deployment not found: {deployment_id}", deployment_id)
                
                running_instances = [i for i in state.instances if i.status == ContainerStatus.RUNNING]
                current_replicas = len(running_instances)
                
                if target_replicas >= current_replicas:
                    logger.info(f"No scale down needed: current={current_replicas}, target={target_replicas}")
                    return True
                
                # 销毁多余实例（从后往前销毁，保留最老的实例）
                instances_to_remove = current_replicas - target_replicas
                for _ in range(instances_to_remove):
                    if running_instances:
                        instance = running_instances.pop()
                        self._destroy_instance(instance)
                        state.instances.remove(instance)
                
                state.desired_replicas = target_replicas
                self._update_deployment_metrics(state)
                
                logger.info(f"Scaled down deployment {deployment_id}: {current_replicas} -> {target_replicas}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to scale down: {e}")
            return False
    
    def scale(self, deployment_id: str, target_replicas: int) -> bool:
        """扩缩容（自动判断方向）
        
        Args:
            deployment_id: 部署ID
            target_replicas: 目标副本数
            
        Returns:
            是否成功
        """
        with self._lock:
            state = self._deployments.get(deployment_id)
            if not state:
                return False
            
            current = len([i for i in state.instances if i.status == ContainerStatus.RUNNING])
            
            if target_replicas > current:
                return self.scale_up(deployment_id, target_replicas)
            elif target_replicas < current:
                return self.scale_down(deployment_id, target_replicas)
            else:
                return True
    
    # ==================== 发布策略操作 ====================
    
    def rolling_update(self, deployment_id: str, target_replicas: int = None, 
                      step: int = 1, new_version: str = None) -> bool:
        """滚动更新
        
        逐步替换旧版本实例为新版本，每次替换 step 个实例。
        
        Args:
            deployment_id: 部署ID
            target_replicas: 目标副本数（可选）
            step: 每次更新的实例数
            new_version: 新版本号
            
        Returns:
            是否成功
        """
        try:
            with self._lock:
                state = self._deployments.get(deployment_id)
                if not state:
                    raise DeploymentNotFoundError(f"Deployment not found: {deployment_id}", deployment_id)
                
                state.release_phase = ReleasePhase.ROLLING
                
                if target_replicas is not None:
                    state.desired_replicas = target_replicas
                
                if new_version:
                    state.previous_version = state.current_version
                    state.current_version = new_version
                
                running_instances = [i for i in state.instances if i.status == ContainerStatus.RUNNING]
                old_version_instances = [i for i in running_instances if i.version != state.current_version]
                
                # 逐步替换旧实例
                instances_to_update = old_version_instances[:step]
                for old_instance in instances_to_update:
                    # 创建新实例
                    new_instance = self._create_instance(deployment_id, version=state.current_version)
                    state.instances.append(new_instance)
                    
                    # 等待新实例就绪后销毁旧实例
                    if self._check_instance_health(new_instance):
                        self._destroy_instance(old_instance)
                        state.instances.remove(old_instance)
                
                # 检查是否还需要更多实例
                current_count = len([i for i in state.instances if i.status == ContainerStatus.RUNNING])
                if current_count < state.desired_replicas:
                    for _ in range(state.desired_replicas - current_count):
                        instance = self._create_instance(deployment_id, version=state.current_version)
                        state.instances.append(instance)
                
                self._update_deployment_metrics(state)
                
                # 检查是否完成
                remaining_old = len([i for i in state.instances 
                                   if i.status == ContainerStatus.RUNNING and i.version != state.current_version])
                if remaining_old == 0:
                    state.release_phase = ReleasePhase.IDLE
                
                logger.info(f"Rolling update on {deployment_id}: step={step}, remaining_old={remaining_old}")
                return True
                
        except Exception as e:
            logger.error(f"Failed rolling update: {e}")
            return False
    
    def canary_release(self, deployment_id: str, percent: int = 10, 
                      new_version: str = None) -> bool:
        """金丝雀发布
        
        将 percent% 的流量切换到新版本。
        
        Args:
            deployment_id: 部署ID
            percent: 金丝雀流量百分比 (0-100)
            new_version: 新版本号
            
        Returns:
            是否成功
        """
        try:
            with self._lock:
                state = self._deployments.get(deployment_id)
                if not state:
                    raise DeploymentNotFoundError(f"Deployment not found: {deployment_id}", deployment_id)
                
                state.release_phase = ReleasePhase.CANARY
                state.canary_percent = max(0, min(100, percent))
                
                if new_version:
                    state.previous_version = state.current_version
                    state.current_version = new_version
                
                running_instances = [i for i in state.instances if i.status == ContainerStatus.RUNNING]
                total_instances = len(running_instances)
                
                if total_instances == 0:
                    logger.warning(f"No running instances for canary release: {deployment_id}")
                    return False
                
                # 计算需要多少个金丝雀实例
                canary_count = max(1, int(total_instances * percent / 100))
                
                # 创建金丝雀实例
                canary_instances = []
                for _ in range(canary_count):
                    instance = self._create_instance(deployment_id, version=state.current_version,
                                                    traffic_weight=percent / canary_count)
                    canary_instances.append(instance)
                    state.instances.append(instance)
                
                # 调整现有实例的流量权重
                main_weight = (100 - percent) / max(1, total_instances)
                for instance in running_instances:
                    if instance.version != state.current_version:
                        instance.traffic_weight = main_weight
                
                self._update_deployment_metrics(state)
                
                logger.info(f"Canary release on {deployment_id}: {percent}% to new version, {canary_count} canary instances")
                return True
                
        except Exception as e:
            logger.error(f"Failed canary release: {e}")
            return False
    
    def promote_canary(self, deployment_id: str) -> bool:
        """提升金丝雀到全量
        
        Args:
            deployment_id: 部署ID
            
        Returns:
            是否成功
        """
        try:
            with self._lock:
                state = self._deployments.get(deployment_id)
                if not state:
                    return False
                
                if state.release_phase != ReleasePhase.CANARY:
                    logger.warning(f"Deployment is not in canary phase: {deployment_id}")
                    return False
                
                # 销毁所有旧版本实例
                old_instances = [i for i in state.instances 
                               if i.status == ContainerStatus.RUNNING and i.version != state.current_version]
                for instance in old_instances:
                    self._destroy_instance(instance)
                    state.instances.remove(instance)
                
                # 扩展金丝雀实例到目标数量
                current_count = len([i for i in state.instances if i.status == ContainerStatus.RUNNING])
                for _ in range(state.desired_replicas - current_count):
                    instance = self._create_instance(deployment_id, version=state.current_version, traffic_weight=100)
                    state.instances.append(instance)
                
                # 重置流量权重
                for instance in state.instances:
                    if instance.status == ContainerStatus.RUNNING:
                        instance.traffic_weight = 100 / state.desired_replicas
                
                state.canary_percent = 100
                state.release_phase = ReleasePhase.IDLE
                self._update_deployment_metrics(state)
                
                logger.info(f"Canary promoted to full release: {deployment_id}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to promote canary: {e}")
            return False
    
    def blue_green_switch(self, deployment_id: str) -> bool:
        """蓝绿切换
        
        即时切换流量从蓝色环境到绿色环境（或反向）。
        
        Args:
            deployment_id: 部署ID
            
        Returns:
            是否成功
        """
        try:
            with self._lock:
                state = self._deployments.get(deployment_id)
                if not state:
                    raise DeploymentNotFoundError(f"Deployment not found: {deployment_id}", deployment_id)
                
                state.release_phase = ReleasePhase.BLUE_GREEN
                
                # 切换活跃环境
                state.blue_active = not state.blue_active
                active_env = "blue" if state.blue_active else "green"
                
                # 更新实例流量权重
                for instance in state.instances:
                    if instance.status == ContainerStatus.RUNNING:
                        # 假设偶数实例是蓝色，奇数实例是绿色
                        is_blue = state.instances.index(instance) % 2 == 0
                        if (is_blue and state.blue_active) or (not is_blue and not state.blue_active):
                            instance.traffic_weight = 100
                        else:
                            instance.traffic_weight = 0
                
                self._update_deployment_metrics(state)
                state.release_phase = ReleasePhase.IDLE
                
                logger.info(f"Blue-green switch on {deployment_id}: active={active_env}")
                return True
                
        except Exception as e:
            logger.error(f"Failed blue-green switch: {e}")
            return False
    
    def blue_green_drain_switch(self, deployment_id: str, drain_timeout: int = 30) -> bool:
        """蓝绿排空切换
        
        等待现有请求处理完成后再切换流量。
        
        Args:
            deployment_id: 部署ID
            drain_timeout: 排空超时时间（秒）
            
        Returns:
            是否成功
        """
        try:
            with self._lock:
                state = self._deployments.get(deployment_id)
                if not state:
                    return False
                
                logger.info(f"Starting drain before blue-green switch: {deployment_id}, timeout={drain_timeout}s")
                
                # 模拟排空等待
                # 在实际实现中，这里应该等待现有连接完成或超时
                time.sleep(min(0.1, drain_timeout / 100))  # 简化模拟
                
                return self.blue_green_switch(deployment_id)
                
        except Exception as e:
            logger.error(f"Failed blue-green drain switch: {e}")
            return False
    
    def ab_test_split(self, deployment_id: str, a_percent: int = 50) -> bool:
        """AB测试分流
        
        将流量按比例分配给 A/B 两个版本。
        
        Args:
            deployment_id: 部署ID
            a_percent: A版本流量百分比 (0-100)
            
        Returns:
            是否成功
        """
        try:
            with self._lock:
                state = self._deployments.get(deployment_id)
                if not state:
                    raise DeploymentNotFoundError(f"Deployment not found: {deployment_id}", deployment_id)
                
                state.release_phase = ReleasePhase.AB_TESTING
                state.ab_test_percent = max(0, min(100, a_percent))
                b_percent = 100 - state.ab_test_percent
                
                running_instances = [i for i in state.instances if i.status == ContainerStatus.RUNNING]
                
                if len(running_instances) < 2:
                    # 需要至少2个实例来做AB测试
                    instance = self._create_instance(deployment_id, version="v2")
                    state.instances.append(instance)
                    running_instances = [i for i in state.instances if i.status == ContainerStatus.RUNNING]
                
                # 分配流量权重
                half = len(running_instances) // 2
                for i, instance in enumerate(running_instances):
                    if i < half:
                        instance.traffic_weight = state.ab_test_percent / max(1, half)
                        instance.version = "v1"  # A版本
                    else:
                        instance.traffic_weight = b_percent / max(1, len(running_instances) - half)
                        instance.version = "v2"  # B版本
                
                self._update_deployment_metrics(state)
                
                logger.info(f"AB test split on {deployment_id}: A={a_percent}%, B={b_percent}%")
                return True
                
        except Exception as e:
            logger.error(f"Failed AB test split: {e}")
            return False
    
    # ==================== 回滚操作 ====================
    
    def rollback(self, deployment_id: str, target_version: str = None) -> bool:
        """回滚部署
        
        回滚到上一个版本或指定版本。
        
        Args:
            deployment_id: 部署ID
            target_version: 目标版本（可选，默认回滚到上一版本）
            
        Returns:
            是否成功
        """
        try:
            with self._lock:
                state = self._deployments.get(deployment_id)
                if not state:
                    raise DeploymentNotFoundError(f"Deployment not found: {deployment_id}", deployment_id)
                
                rollback_version = target_version or state.previous_version
                if not rollback_version:
                    rollback_version = "v1"  # 默认回滚到v1
                
                logger.info(f"Rolling back {deployment_id} to version {rollback_version}")
                
                # 销毁当前版本实例
                current_instances = [i for i in state.instances if i.status == ContainerStatus.RUNNING]
                for instance in current_instances:
                    self._destroy_instance(instance)
                    state.instances.remove(instance)
                
                # 创建回滚版本的实例
                for _ in range(state.desired_replicas):
                    instance = self._create_instance(deployment_id, version=rollback_version)
                    state.instances.append(instance)
                
                # 更新版本信息
                state.previous_version = state.current_version
                state.current_version = rollback_version
                state.release_phase = ReleasePhase.IDLE
                state.canary_percent = 0
                
                self._update_deployment_metrics(state)
                
                logger.info(f"Rollback completed: {deployment_id} -> {rollback_version}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to rollback: {e}")
            return False
    
    # ==================== 健康检查 ====================
    
    def health_check(self, deployment_id: str) -> Dict[str, Any]:
        """执行健康检查
        
        Args:
            deployment_id: 部署ID
            
        Returns:
            健康检查结果
        """
        with self._lock:
            state = self._deployments.get(deployment_id)
            if not state:
                return {'status': 'not_found', 'healthy': False, 'instances': []}
            
            results = []
            healthy_count = 0
            
            for instance in state.instances:
                if instance.status == ContainerStatus.RUNNING:
                    is_healthy = self._check_instance_health(instance)
                    if is_healthy:
                        healthy_count += 1
                    results.append({
                        'instance_id': instance.instance_id,
                        'status': instance.status.value,
                        'healthy': is_healthy,
                        'version': instance.version
                    })
            
            total = len([i for i in state.instances if i.status == ContainerStatus.RUNNING])
            
            return {
                'deployment_id': deployment_id,
                'status': 'healthy' if healthy_count == total and total > 0 else 'unhealthy',
                'healthy': healthy_count == total and total > 0,
                'healthy_count': healthy_count,
                'total_count': total,
                'instances': results
            }
    
    # ==================== 统计和监控 ====================
    
    def get_metrics(self, deployment_id: str) -> Dict[str, Any]:
        """获取部署指标
        
        Args:
            deployment_id: 部署ID
            
        Returns:
            部署指标
        """
        with self._lock:
            state = self._deployments.get(deployment_id)
            if not state:
                return {}
            
            instances = [i for i in state.instances if i.status == ContainerStatus.RUNNING]
            
            total_requests = sum(i.request_count for i in instances)
            total_errors = sum(i.error_count for i in instances)
            avg_cpu = sum(i.cpu_usage for i in instances) / len(instances) if instances else 0
            avg_memory = sum(i.memory_usage for i in instances) / len(instances) if instances else 0
            
            return {
                'deployment_id': deployment_id,
                'status': state.status.value,
                'replicas': {
                    'desired': state.desired_replicas,
                    'available': state.available_replicas,
                    'ready': state.ready_replicas
                },
                'requests': {
                    'total': total_requests,
                    'errors': total_errors,
                    'error_rate': total_errors / total_requests if total_requests > 0 else 0
                },
                'resources': {
                    'avg_cpu_percent': round(avg_cpu, 2),
                    'avg_memory_percent': round(avg_memory, 2)
                },
                'release': {
                    'phase': state.release_phase.value,
                    'current_version': state.current_version,
                    'previous_version': state.previous_version
                }
            }


# 全局容器管理器实例
_container_manager = None
_manager_lock = threading.Lock()


def get_container_manager() -> ContainerManager:
    """获取容器管理器实例（线程安全单例）"""
    global _container_manager
    
    if _container_manager is None:
        with _manager_lock:
            if _container_manager is None:
                _container_manager = ContainerManager()
    
    return _container_manager


def reset_container_manager():
    """重置容器管理器（用于测试）"""
    global _container_manager
    with _manager_lock:
        _container_manager = None

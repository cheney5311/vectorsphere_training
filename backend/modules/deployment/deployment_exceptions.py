"""部署相关异常定义"""


class DeploymentError(Exception):
    """部署错误基类"""
    
    def __init__(self, message: str, deployment_id: str = None):
        super().__init__(message)
        self.deployment_id = deployment_id
        self.message = message
    
    def __str__(self):
        if self.deployment_id:
            return f"{self.message} (deployment_id={self.deployment_id})"
        return self.message


class ContainerDeploymentError(DeploymentError):
    """容器部署错误"""
    pass


class DeploymentConfigError(DeploymentError):
    """部署配置错误"""
    pass


class DeploymentNotFoundError(DeploymentError):
    """部署未找到错误"""
    pass


class ScalingError(DeploymentError):
    """扩缩容错误"""
    pass


class ReleaseStrategyError(DeploymentError):
    """发布策略错误"""
    pass


class RollbackError(DeploymentError):
    """回滚错误"""
    pass


class HealthCheckError(DeploymentError):
    """健康检查错误"""
    pass
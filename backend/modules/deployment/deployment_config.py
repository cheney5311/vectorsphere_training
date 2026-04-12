"""部署配置类"""

from dataclasses import dataclass
from typing import Dict, Any, Optional


@dataclass
class DeploymentConfig:
    """部署配置"""
    
    model_id: str
    deployment_mode: str = "online"  # online, batch, edge
    replicas: int = 1
    cpu_limit: str = "1000m"
    memory_limit: str = "2Gi"
    gpu_limit: int = 0
    port: int = 8080
    health_check_path: str = "/health"
    environment_vars: Optional[Dict[str, str]] = None
    custom_config: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.environment_vars is None:
            self.environment_vars = {}
        if self.custom_config is None:
            self.custom_config = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'model_id': self.model_id,
            'deployment_mode': self.deployment_mode,
            'replicas': self.replicas,
            'cpu_limit': self.cpu_limit,
            'memory_limit': self.memory_limit,
            'gpu_limit': self.gpu_limit,
            'port': self.port,
            'health_check_path': self.health_check_path,
            'environment_vars': self.environment_vars,
            'custom_config': self.custom_config
        }
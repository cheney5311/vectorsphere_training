"""部署管理器"""

import logging
from typing import Dict, Any, Optional, List
from .deployment_config import DeploymentConfig
from .container_manager import get_container_manager
from .deployment_exceptions import DeploymentError, DeploymentNotFoundError

logger = logging.getLogger(__name__)


class DeploymentManager:
    """部署管理器"""
    
    def __init__(self):
        self.container_manager = get_container_manager()
        self.deployments: Dict[str, Dict[str, Any]] = {}
        
    def create_deployment(self, deployment_id: str, config: DeploymentConfig) -> Dict[str, Any]:
        """创建部署
        
        Args:
            deployment_id: 部署ID
            config: 部署配置
            
        Returns:
            部署信息
            
        Raises:
            DeploymentError: 部署失败
        """
        try:
            logger.info(f"创建部署: {deployment_id}")
            
            # 部署容器
            container_info = self.container_manager.deploy_container(deployment_id, config)
            
            # 保存部署信息
            deployment_info = {
                'deployment_id': deployment_id,
                'model_id': config.model_id,
                'deployment_mode': config.deployment_mode,
                'status': 'active',
                'container_info': container_info,
                'config': config.to_dict(),
                'created_at': '2024-01-01T00:00:00Z'
            }
            
            self.deployments[deployment_id] = deployment_info
            
            logger.info(f"部署创建成功: {deployment_id}")
            return deployment_info
            
        except Exception as e:
            logger.error(f"创建部署失败: {e}")
            raise DeploymentError(f"创建部署失败: {str(e)}", deployment_id)
    
    def get_deployment(self, deployment_id: str) -> Optional[Dict[str, Any]]:
        """获取部署信息
        
        Args:
            deployment_id: 部署ID
            
        Returns:
            部署信息
        """
        return self.deployments.get(deployment_id)
    
    def get_deployment_status(self, deployment_id: str) -> Dict[str, Any]:
        """获取部署状态
        
        Args:
            deployment_id: 部署ID
            
        Returns:
            部署状态
            
        Raises:
            DeploymentNotFoundError: 部署未找到
        """
        deployment = self.get_deployment(deployment_id)
        if not deployment:
            raise DeploymentNotFoundError(f"部署未找到: {deployment_id}", deployment_id)
        
        # 获取容器状态
        container_status = self.container_manager.get_container_status(deployment_id)
        
        return {
            'deployment_id': deployment_id,
            'status': deployment.get('status', 'unknown'),
            'container_status': container_status,
            'model_id': deployment.get('model_id'),
            'deployment_mode': deployment.get('deployment_mode')
        }
    
    def delete_deployment(self, deployment_id: str) -> bool:
        """删除部署
        
        Args:
            deployment_id: 部署ID
            
        Returns:
            是否成功删除
        """
        try:
            # 停止容器
            self.container_manager.stop_container(deployment_id)
            
            # 删除部署记录
            if deployment_id in self.deployments:
                del self.deployments[deployment_id]
                logger.info(f"部署已删除: {deployment_id}")
                return True
            return False
        except Exception as e:
            logger.error(f"删除部署失败: {e}")
            return False
    
    def list_deployments(self) -> List[Dict[str, Any]]:
        """列出所有部署
        
        Returns:
            部署列表
        """
        return list(self.deployments.values())


# 全局部署管理器实例
_deployment_manager = None


def get_deployment_manager() -> DeploymentManager:
    """获取部署管理器实例"""
    global _deployment_manager
    if _deployment_manager is None:
        _deployment_manager = DeploymentManager()
    return _deployment_manager
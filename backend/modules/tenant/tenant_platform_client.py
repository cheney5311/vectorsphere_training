"""租户平台客户端

用于与VectorSphere租户平台进行通信，上报训练进度等信息。
"""

import os
import json
import logging
from typing import Dict, Any, Optional
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# 添加导入
from backend.modules.training.config.tenant_platform_config import get_tenant_platform_config

logger = logging.getLogger(__name__)


class TenantPlatformClient:
    """租户平台客户端"""
    
    def __init__(self, base_url: Optional[str] = None, api_token: Optional[str] = None):
        """初始化租户平台客户端
        
        Args:
            base_url: 租户平台基础URL，如果未提供则从环境变量获取
            api_token: API令牌，如果未提供则从环境变量获取
        """
        # 获取配置
        config = get_tenant_platform_config()
        
        self.base_url = base_url or os.getenv('TENANT_PLATFORM_URL') or config.url
        self.api_token = api_token or os.getenv('TENANT_PLATFORM_API_TOKEN') or config.api_token
        
        # 创建带重试机制的会话
        self.session = requests.Session()
        retry_strategy = Retry(
            total=config.retries,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # 设置默认请求头
        self.session.headers.update({
            'Content-Type': 'application/json',
            'User-Agent': 'VectorSphere-Intelligent-Platform/1.0'
        })
        
        if self.api_token:
            self.session.headers.update({
                'Authorization': f'Bearer {self.api_token}'
            })
    
    def report_training_progress(self, job_id: str, progress_data: Dict[str, Any]) -> bool:
        """向租户平台上报训练进度
        
        Args:
            job_id: 训练任务ID
            progress_data: 进度数据
            
        Returns:
            是否上报成功
        """
        try:
            url = f"{self.base_url}/api/v1/training/jobs/{job_id}/progress"
            
            # 准备请求数据
            payload = {
                "progress": progress_data.get("progress", 0),
                "metrics": progress_data.get("metrics", {}),
                "status": progress_data.get("status", "running")
            }
            
            # 如果有错误信息，也包含在请求中
            if "error" in progress_data:
                payload["error"] = progress_data["error"]
            
            response = self.session.put(url, json=payload, timeout=30)
            response.raise_for_status()
            
            logger.info(f"成功向租户平台上报训练进度: job_id={job_id}, progress={payload['progress']}")
            return True
            
        except requests.exceptions.RequestException as e:
            logger.error(f"向租户平台上报训练进度失败: job_id={job_id}, error={str(e)}")
            return False
        except Exception as e:
            logger.error(f"向租户平台上报训练进度时发生未知错误: job_id={job_id}, error={str(e)}")
            return False
    
    def report_training_status(self, job_id: str, status: str, error_message: Optional[str] = None) -> bool:
        """向租户平台上报训练状态
        
        Args:
            job_id: 训练任务ID
            status: 状态
            error_message: 错误信息（可选）
            
        Returns:
            是否上报成功
        """
        try:
            url = f"{self.base_url}/api/v1/training/jobs/{job_id}/status"
            
            # 准备请求数据
            payload = {
                "status": status
            }
            
            if error_message:
                payload["error"] = error_message
            
            response = self.session.put(url, json=payload, timeout=30)
            response.raise_for_status()
            
            logger.info(f"成功向租户平台上报训练状态: job_id={job_id}, status={status}")
            return True
            
        except requests.exceptions.RequestException as e:
            logger.error(f"向租户平台上报训练状态失败: job_id={job_id}, error={str(e)}")
            return False
        except Exception as e:
            logger.error(f"向租户平台上报训练状态时发生未知错误: job_id={job_id}, error={str(e)}")
            return False
    
    def get_training_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """从租户平台获取训练任务信息
        
        Args:
            job_id: 训练任务ID
            
        Returns:
            训练任务信息，如果获取失败则返回None
        """
        try:
            url = f"{self.base_url}/api/v1/training/jobs/{job_id}"
            
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            logger.info(f"成功从租户平台获取训练任务信息: job_id={job_id}")
            return data.get("data") if isinstance(data, dict) and "data" in data else data
            
        except requests.exceptions.RequestException as e:
            logger.error(f"从租户平台获取训练任务信息失败: job_id={job_id}, error={str(e)}")
            return None
        except Exception as e:
            logger.error(f"从租户平台获取训练任务信息时发生未知错误: job_id={job_id}, error={str(e)}")
            return None


# 全局租户平台客户端实例
_global_tenant_client: Optional[TenantPlatformClient] = None


def get_tenant_platform_client() -> TenantPlatformClient:
    """获取全局租户平台客户端实例
    
    Returns:
        TenantPlatformClient: 租户平台客户端实例
    """
    global _global_tenant_client
    
    if _global_tenant_client is None:
        _global_tenant_client = TenantPlatformClient()
        
    return _global_tenant_client


def report_training_progress_to_tenant_platform(job_id: str, progress_data: Dict[str, Any]) -> bool:
    """向租户平台上报训练进度的便捷函数
    
    Args:
        job_id: 训练任务ID
        progress_data: 进度数据
        
    Returns:
        是否上报成功
    """
    # 环境变量禁用优先
    if os.environ.get("TENANT_PLATFORM_DISABLED") == "1":
        logger.debug("租户平台上报已禁用，跳过进度上报")
        return True
    # 检查是否启用租户平台集成
    config = get_tenant_platform_config()
    if not config.enabled:
        logger.debug("租户平台集成未启用，跳过进度上报")
        return True
    
    client = get_tenant_platform_client()
    return client.report_training_progress(job_id, progress_data)


def report_training_status_to_tenant_platform(job_id: str, status: str, error_message: Optional[str] = None) -> bool:
    """向租户平台上报训练状态的便捷函数
    
    Args:
        job_id: 训练任务ID
        status: 状态
        error_message: 错误信息（可选）
        
    Returns:
        是否上报成功
    """
    # 环境变量禁用优先
    if os.environ.get("TENANT_PLATFORM_DISABLED") == "1":
        logger.debug("租户平台上报已禁用，跳过状态上报")
        return True
    # 检查是否启用租户平台集成
    config = get_tenant_platform_config()
    if not config.enabled:
        logger.debug("租户平台集成未启用，跳过状态上报")
        return True
    
    client = get_tenant_platform_client()
    return client.report_training_status(job_id, status, error_message)
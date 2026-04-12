"""租户平台集成

提供与租户平台的集成功能，包括进度报告和状态同步。
"""

import logging
import requests
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


def report_training_progress_to_tenant_platform(
    session_id: str,
    progress_data: Dict[str, Any],
    tenant_config: Optional[Dict[str, Any]] = None
) -> bool:
    """向租户平台报告训练进度
    
    Args:
        session_id: 训练会话ID
        progress_data: 进度数据
        tenant_config: 租户配置
        
    Returns:
        bool: 报告是否成功
    """
    try:
        # 如果没有租户配置，则跳过报告
        if not tenant_config or not tenant_config.get('enabled', False):
            logger.debug(f"租户平台集成未启用，跳过进度报告: {session_id}")
            return True
            
        # 构建报告数据
        report_data = {
            'session_id': session_id,
            'timestamp': datetime.now().isoformat(),
            'progress': progress_data
        }
        
        # 发送到租户平台
        endpoint = tenant_config.get('progress_endpoint')
        if endpoint:
            headers = {
                'Content-Type': 'application/json',
                'Authorization': f"Bearer {tenant_config.get('api_token', '')}"
            }
            
            response = requests.post(
                endpoint,
                json=report_data,
                headers=headers,
                timeout=30
            )
            
            if response.status_code == 200:
                logger.info(f"成功向租户平台报告训练进度: {session_id}")
                return True
            else:
                logger.warning(f"向租户平台报告训练进度失败: {session_id}, 状态码: {response.status_code}")
                return False
        else:
            logger.warning(f"租户平台进度端点未配置: {session_id}")
            return False
            
    except Exception as e:
        logger.error(f"向租户平台报告训练进度时发生错误: {session_id}, 错误: {e}")
        return False


def report_training_status_to_tenant_platform(
    session_id: str,
    status: str,
    status_data: Optional[Dict[str, Any]] = None,
    tenant_config: Optional[Dict[str, Any]] = None
) -> bool:
    """向租户平台报告训练状态
    
    Args:
        session_id: 训练会话ID
        status: 训练状态
        status_data: 状态数据
        tenant_config: 租户配置
        
    Returns:
        bool: 报告是否成功
    """
    try:
        # 如果没有租户配置，则跳过报告
        if not tenant_config or not tenant_config.get('enabled', False):
            logger.debug(f"租户平台集成未启用，跳过状态报告: {session_id}")
            return True
            
        # 构建报告数据
        report_data = {
            'session_id': session_id,
            'timestamp': datetime.now().isoformat(),
            'status': status,
            'data': status_data or {}
        }
        
        # 发送到租户平台
        endpoint = tenant_config.get('status_endpoint')
        if endpoint:
            headers = {
                'Content-Type': 'application/json',
                'Authorization': f"Bearer {tenant_config.get('api_token', '')}"
            }
            
            response = requests.post(
                endpoint,
                json=report_data,
                headers=headers,
                timeout=30
            )
            
            if response.status_code == 200:
                logger.info(f"成功向租户平台报告训练状态: {session_id}, 状态: {status}")
                return True
            else:
                logger.warning(f"向租户平台报告训练状态失败: {session_id}, 状态码: {response.status_code}")
                return False
        else:
            logger.warning(f"租户平台状态端点未配置: {session_id}")
            return False
            
    except Exception as e:
        logger.error(f"向租户平台报告训练状态时发生错误: {session_id}, 错误: {e}")
        return False
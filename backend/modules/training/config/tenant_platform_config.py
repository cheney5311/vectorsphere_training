"""租户平台配置管理

用于加载和管理与租户平台集成相关的配置。
"""

import os
from typing import Optional, Dict, Any
import yaml
from pathlib import Path

# 默认配置
DEFAULT_TENANT_PLATFORM_CONFIG = {
    "enabled": True,
    "url": "http://tenant-platform:8080",
    "api_token": "",
    "timeout": 30,
    "retries": 3
}


class TenantPlatformConfig:
    """租户平台配置管理器"""
    
    def __init__(self, config_file: Optional[str] = None):
        """初始化租户平台配置管理器
        
        Args:
            config_file: 配置文件路径，如果未提供则使用默认路径
        """
        self.config_file = config_file
        self._config = DEFAULT_TENANT_PLATFORM_CONFIG.copy()
        self._load_config()
    
    def _load_config(self):
        """加载配置"""
        # 1. 从配置文件加载
        if self.config_file and os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config_data = yaml.safe_load(f)
                
                if isinstance(config_data, dict) and 'tenant_platform' in config_data:
                    self._config.update(config_data['tenant_platform'])
            except Exception as e:
                print(f"警告: 从配置文件加载租户平台配置失败: {e}")
        
        # 2. 从环境变量覆盖
        self._config['enabled'] = self._get_env_bool('TENANT_PLATFORM_ENABLED', self._config['enabled'])
        self._config['url'] = os.getenv('TENANT_PLATFORM_URL', self._config['url'])
        self._config['api_token'] = os.getenv('TENANT_PLATFORM_API_TOKEN', self._config['api_token'])
        self._config['timeout'] = int(os.getenv('TENANT_PLATFORM_TIMEOUT', str(self._config['timeout'])))
        self._config['retries'] = int(os.getenv('TENANT_PLATFORM_RETRIES', str(self._config['retries'])))
    
    def _get_env_bool(self, key: str, default: bool) -> bool:
        """从环境变量获取布尔值"""
        value = os.getenv(key, str(default)).lower()
        return value in ('true', '1', 'yes', 'on')
    
    @property
    def enabled(self) -> bool:
        """是否启用租户平台集成"""
        return self._config['enabled']
    
    @property
    def url(self) -> str:
        """租户平台URL"""
        return self._config['url']
    
    @property
    def api_token(self) -> str:
        """API令牌"""
        return self._config['api_token']
    
    @property
    def timeout(self) -> int:
        """请求超时时间（秒）"""
        return self._config['timeout']
    
    @property
    def retries(self) -> int:
        """重试次数"""
        return self._config['retries']
    
    def get_config(self) -> Dict[str, Any]:
        """获取完整配置"""
        return self._config.copy()


# 全局配置实例
_global_tenant_config: Optional[TenantPlatformConfig] = None


def get_tenant_platform_config() -> TenantPlatformConfig:
    """获取全局租户平台配置实例
    
    Returns:
        TenantPlatformConfig: 租户平台配置实例
    """
    global _global_tenant_config
    
    if _global_tenant_config is None:
        # 尝试从项目配置文件加载
        config_file = None
        project_root = Path(__file__).parent.parent.parent.parent
        config_path = project_root / "config" / "config.yaml"
        if config_path.exists():
            config_file = str(config_path)
        
        _global_tenant_config = TenantPlatformConfig(config_file)
        
    return _global_tenant_config
"""
API 配置管理模块
用于管理各种 AI API 服务的配置，包括 ChatGPT、DeepSeek 等
支持安全的 token 存储、配置验证和动态配置更新
"""

import os
import json
import logging
from enum import Enum
from dataclasses import dataclass, asdict
from typing import Dict, Optional, List, Any
from cryptography.fernet import Fernet
import base64

logger = logging.getLogger(__name__)


class APIProvider(Enum):
    """API 提供商枚举"""
    OPENAI = "openai"
    DEEPSEEK = "deepseek"
    LOCAL = "local"


@dataclass
class APIConfig:
    """API 配置数据类"""
    provider: str
    api_key: str
    base_url: str
    model_name: str
    max_tokens: int = 4096
    temperature: float = 0.7
    timeout: int = 30
    retry_attempts: int = 3
    enabled: bool = True
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class APIConfigManager:
    """API 配置管理器"""
    
    def __init__(self, config_file: str = "api_configs.json", encryption_key: str = None):
        self.config_file = config_file
        self.configs: Dict[str, APIConfig] = {}
        
        # 初始化加密密钥
        if encryption_key:
            self.encryption_key = encryption_key.encode()
        else:
            # 从环境变量获取或生成新的加密密钥
            key_env = os.getenv('API_CONFIG_ENCRYPTION_KEY')
            if key_env:
                self.encryption_key = base64.urlsafe_b64decode(key_env.encode())
            else:
                self.encryption_key = Fernet.generate_key()
                logger.warning("生成了新的加密密钥，请保存到环境变量 API_CONFIG_ENCRYPTION_KEY")
        
        self.cipher = Fernet(self.encryption_key)
        
        # 加载配置
        self.load_configs()
        
        # 设置默认配置
        self._setup_default_configs()
    
    def _encrypt_data(self, data: str) -> str:
        """加密数据"""
        try:
            return self.cipher.encrypt(data.encode()).decode()
        except Exception as e:
            logger.error(f"数据加密失败: {e}")
            return data
    
    def _decrypt_data(self, encrypted_data: str) -> str:
        """解密数据"""
        try:
            return self.cipher.decrypt(encrypted_data.encode()).decode()
        except Exception as e:
            logger.error(f"数据解密失败: {e}")
            return encrypted_data
    
    def _setup_default_configs(self):
        """设置默认配置"""
        default_configs = {
            APIProvider.OPENAI.value: APIConfig(
                provider=APIProvider.OPENAI.value,
                api_key=os.getenv('OPENAI_API_KEY', ''),
                base_url="https://api.openai.com/v1",
                model_name="gpt-3.5-turbo",
                max_tokens=4096,
                temperature=0.7,
                timeout=30,
                retry_attempts=3,
                enabled=bool(os.getenv('OPENAI_API_KEY')),
                metadata={
                    "description": "OpenAI ChatGPT API",
                    "pricing_model": "token_based",
                    "supported_models": ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"]
                }
            ),
            APIProvider.DEEPSEEK.value: APIConfig(
                provider=APIProvider.DEEPSEEK.value,
                api_key=os.getenv('DEEPSEEK_API_KEY', ''),
                base_url="https://api.deepseek.com/v1",
                model_name="deepseek-chat",
                max_tokens=4096,
                temperature=0.7,
                timeout=30,
                retry_attempts=3,
                enabled=bool(os.getenv('DEEPSEEK_API_KEY')),
                metadata={
                    "description": "DeepSeek API",
                    "pricing_model": "token_based",
                    "supported_models": ["deepseek-chat", "deepseek-coder"]
                }
            ),
            APIProvider.LOCAL.value: APIConfig(
                provider=APIProvider.LOCAL.value,
                api_key="",
                base_url="http://localhost:11434",
                model_name="llama2",
                max_tokens=4096,
                temperature=0.7,
                timeout=60,
                retry_attempts=2,
                enabled=True,
                metadata={
                    "description": "本地模型服务 (Ollama)",
                    "pricing_model": "free",
                    "supported_models": ["llama2", "codellama", "mistral"]
                }
            )
        }
        
        # 只添加不存在的默认配置
        for provider, config in default_configs.items():
            if provider not in self.configs:
                self.configs[provider] = config
    
    def load_configs(self):
        """从文件加载配置"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                for provider, config_data in data.items():
                    # 解密敏感数据
                    if 'api_key' in config_data and config_data['api_key']:
                        config_data['api_key'] = self._decrypt_data(config_data['api_key'])
                    
                    self.configs[provider] = APIConfig(**config_data)
                
                logger.info(f"成功加载 {len(self.configs)} 个 API 配置")
        except Exception as e:
            logger.error(f"加载配置文件失败: {e}")
            self.configs = {}
    
    def save_configs(self):
        """保存配置到文件"""
        try:
            data = {}
            for provider, config in self.configs.items():
                config_dict = asdict(config)
                # 加密敏感数据
                if config_dict['api_key']:
                    config_dict['api_key'] = self._encrypt_data(config_dict['api_key'])
                data[provider] = config_dict
            
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.info("配置文件保存成功")
        except Exception as e:
            logger.error(f"保存配置文件失败: {e}")
    
    def add_config(self, provider: str, config: APIConfig):
        """添加新的 API 配置"""
        self.configs[provider] = config
        self.save_configs()
        logger.info(f"添加 API 配置: {provider}")
    
    def get_config(self, provider: str) -> Optional[APIConfig]:
        """获取指定提供商的配置"""
        return self.configs.get(provider)
    
    def update_config(self, provider: str, **kwargs):
        """更新指定提供商的配置"""
        if provider in self.configs:
            config = self.configs[provider]
            for key, value in kwargs.items():
                if hasattr(config, key):
                    setattr(config, key, value)
            self.save_configs()
            logger.info(f"更新 API 配置: {provider}")
        else:
            logger.warning(f"配置不存在: {provider}")
    
    def remove_config(self, provider: str):
        """删除指定提供商的配置"""
        if provider in self.configs:
            del self.configs[provider]
            self.save_configs()
            logger.info(f"删除 API 配置: {provider}")
    
    def list_configs(self) -> List[str]:
        """列出所有配置的提供商"""
        return list(self.configs.keys())
    
    def get_enabled_configs(self) -> Dict[str, APIConfig]:
        """获取所有启用的配置"""
        return {k: v for k, v in self.configs.items() if v.enabled}
    
    def validate_config(self, provider: str) -> bool:
        """验证指定提供商的配置"""
        config = self.get_config(provider)
        if not config:
            return False
        
        # 基本验证
        if not config.api_key and provider != APIProvider.LOCAL.value:
            logger.warning(f"{provider} 缺少 API 密钥")
            return False
        
        if not config.base_url:
            logger.warning(f"{provider} 缺少基础 URL")
            return False
        
        if not config.model_name:
            logger.warning(f"{provider} 缺少模型名称")
            return False
        
        return True
    
    def get_encryption_key_base64(self) -> str:
        """获取 base64 编码的加密密钥"""
        return base64.urlsafe_b64encode(self.encryption_key).decode()


# 全局配置管理器实例
api_config_manager = APIConfigManager()
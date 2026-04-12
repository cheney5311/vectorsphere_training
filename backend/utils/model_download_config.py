"""
模型下载配置模块
配置国内镜像源和下载参数，解决模型下载超时问题
"""

import os
import logging
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class ModelDownloadConfig:
    """模型下载配置管理器"""
    
    def __init__(self):
        self.setup_environment()
        self.setup_cache_directories()
        
    def setup_environment(self):
        """设置环境变量"""
        # Hugging Face 镜像配置
        hf_endpoint = os.getenv('HF_ENDPOINT', 'https://hf-mirror.com')
        os.environ['HF_ENDPOINT'] = hf_endpoint
        
        # 缓存目录配置
        hf_cache = os.getenv('HF_HUB_CACHE', '/tmp/huggingface_cache')
        hf_datasets_cache = os.getenv('HF_DATASETS_CACHE', '/tmp/huggingface_datasets')
        modelscope_cache = os.getenv('MODELSCOPE_CACHE', '/tmp/modelscope_cache')
        
        os.environ['HF_HUB_CACHE'] = hf_cache
        os.environ['HF_DATASETS_CACHE'] = hf_datasets_cache
        os.environ['MODELSCOPE_CACHE'] = modelscope_cache
        
        # 设置代理（如果配置了）
        http_proxy = os.getenv('HTTP_PROXY')
        https_proxy = os.getenv('HTTPS_PROXY')
        if http_proxy:
            os.environ['HTTP_PROXY'] = http_proxy
        if https_proxy:
            os.environ['HTTPS_PROXY'] = https_proxy
            
        logger.info(f"模型下载配置已设置: HF_ENDPOINT={hf_endpoint}")
        
    def setup_cache_directories(self):
        """创建缓存目录"""
        cache_dirs = [
            os.getenv('HF_HUB_CACHE', '/tmp/huggingface_cache'),
            os.getenv('HF_DATASETS_CACHE', '/tmp/huggingface_datasets'),
            os.getenv('MODELSCOPE_CACHE', '/tmp/modelscope_cache')
        ]
        
        for cache_dir in cache_dirs:
            Path(cache_dir).mkdir(parents=True, exist_ok=True)
            logger.debug(f"缓存目录已创建: {cache_dir}")
    
    def get_download_config(self) -> Dict[str, Any]:
        """获取下载配置"""
        return {
            'timeout': int(os.getenv('MODEL_DOWNLOAD_TIMEOUT', '1800')),
            'retries': int(os.getenv('MODEL_DOWNLOAD_RETRIES', '3')),
            'retry_delay': int(os.getenv('MODEL_DOWNLOAD_RETRY_DELAY', '10')),
            'hf_endpoint': os.getenv('HF_ENDPOINT', 'https://hf-mirror.com'),
            'modelscope_endpoint': os.getenv('MODELSCOPE_ENDPOINT', 'https://modelscope.cn'),
            'cache_dir': os.getenv('HF_HUB_CACHE', '/tmp/huggingface_cache')
        }
    
    def download_model_with_retry(self, model_name: str, **kwargs) -> Optional[str]:
        """
        使用重试机制下载模型
        
        Args:
            model_name: 模型名称
            **kwargs: 其他下载参数
            
        Returns:
            模型路径或None
        """
        import time
        from transformers import AutoModel, AutoTokenizer
        
        config = self.get_download_config()
        
        for attempt in range(config['retries']):
            try:
                logger.info(f"尝试下载模型 {model_name} (第{attempt + 1}次)")
                
                # 设置下载参数
                download_kwargs = {
                    'cache_dir': config['cache_dir'],
                    'local_files_only': False,
                    'force_download': False,
                    **kwargs
                }
                
                # 下载模型和tokenizer
                model = AutoModel.from_pretrained(model_name, **download_kwargs)
                tokenizer = AutoTokenizer.from_pretrained(model_name, **download_kwargs)
                
                logger.info(f"模型 {model_name} 下载成功")
                return config['cache_dir']
                
            except Exception as e:
                logger.warning(f"模型下载失败 (第{attempt + 1}次): {str(e)}")
                if attempt < config['retries'] - 1:
                    logger.info(f"等待 {config['retry_delay']} 秒后重试...")
                    time.sleep(config['retry_delay'])
                else:
                    logger.error(f"模型 {model_name} 下载失败，已达到最大重试次数")
                    
        return None
    
    def setup_pytorch_mirror(self):
        """设置PyTorch国内镜像源"""
        pytorch_index = os.getenv('PYTORCH_INDEX_URL', 'https://pypi.tuna.tsinghua.edu.cn/simple/')
        pytorch_extra = os.getenv('PYTORCH_EXTRA_INDEX_URL', 'https://download.pytorch.org/whl/cpu')
        
        # 这些配置主要用于pip安装时
        os.environ['PIP_INDEX_URL'] = pytorch_index
        os.environ['PIP_EXTRA_INDEX_URL'] = pytorch_extra
        
        logger.info(f"PyTorch镜像源已配置: {pytorch_index}")
    
    def get_modelscope_config(self) -> Dict[str, str]:
        """获取ModelScope配置"""
        return {
            'endpoint': os.getenv('MODELSCOPE_ENDPOINT', 'https://modelscope.cn'),
            'cache_dir': os.getenv('MODELSCOPE_CACHE', '/tmp/modelscope_cache')
        }

# 全局配置实例
model_download_config = ModelDownloadConfig()

def setup_model_download_environment():
    """设置模型下载环境（供其他模块调用）"""
    model_download_config.setup_environment()
    model_download_config.setup_pytorch_mirror()
    logger.info("模型下载环境配置完成")

def get_model_download_config() -> Dict[str, Any]:
    """获取模型下载配置（供其他模块调用）"""
    return model_download_config.get_download_config()

def download_model_safely(model_name: str, **kwargs) -> Optional[str]:
    """安全下载模型（供其他模块调用）"""
    return model_download_config.download_model_with_retry(model_name, **kwargs)
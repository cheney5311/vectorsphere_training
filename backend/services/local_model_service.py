"""本地模型管理服务

提供本地模型的加载、管理和调用功能，支持多种模型类型。
现已扩展支持云端 API 调用，包括 ChatGPT 和 DeepSeek。
"""

import asyncio
import json
import logging
import os
from datetime import datetime
from typing import Dict, Any, List, Optional, Union

import ollama
import openai
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# 导入云端 API 客户端
from .chatgpt_api_client import chatgpt_client, ChatMessage
from .deepseek_api_client import deepseek_client, DeepSeekMessage

logger = logging.getLogger(__name__)


class ModelConfig:
    """模型配置类"""
    
    def __init__(self, 
                 model_name: str,
                 model_type: str,
                 model_path: Optional[str] = None,
                 api_key: Optional[str] = None,
                 api_base: Optional[str] = None,
                 max_tokens: int = 2048,
                 temperature: float = 0.7,
                 **kwargs):
        self.model_name = model_name
        self.model_type = model_type  # ollama, transformers, openai, anthropic, chatgpt, deepseek
        self.model_path = model_path
        self.api_key = api_key
        self.api_base = api_base
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.kwargs = kwargs


class LocalModelService:
    """本地模型服务"""
    
    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.model_configs: Dict[str, ModelConfig] = {}
        self.default_model = None
        self._initialize_default_models()
    
    def _initialize_default_models(self):
        """初始化默认模型配置"""
        # Ollama 模型配置
        self.register_model(ModelConfig(
            model_name="llama2",
            model_type="ollama",
            max_tokens=2048,
            temperature=0.7
        ))

        self.register_model(ModelConfig(
            model_name="mistral",
            model_type="ollama",
            max_tokens=2048,
            temperature=0.7
        ))
        
        # Transformers 模型配置
        self.register_model(ModelConfig(
            model_name="microsoft/DialoGPT-medium",
            model_type="transformers",
            max_tokens=1024,
            temperature=0.8
        ))
        
        # OpenAI 模型配置
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            self.register_model(ModelConfig(
                model_name="gpt-3.5-turbo",
                model_type="openai",
                api_key=api_key,
                max_tokens=2048,
                temperature=0.7
            ))
        
        # 云端 API 模型配置
        # ChatGPT 模型配置
        if chatgpt_client.is_available():
            self.register_model(ModelConfig(
                model_name="chatgpt-3.5-turbo",
                model_type="chatgpt",
                max_tokens=4096,
                temperature=0.7
            ))

            self.register_model(ModelConfig(
                model_name="chatgpt-4",
                model_type="chatgpt",
                max_tokens=4096,
                temperature=0.7
            ))

        # DeepSeek 模型配置
        if deepseek_client.is_available():
            self.register_model(ModelConfig(
                model_name="deepseek-chat",
                model_type="deepseek",
                max_tokens=4096,
                temperature=0.7
            ))

            self.register_model(ModelConfig(
                model_name="deepseek-coder",
                model_type="deepseek",
                max_tokens=4096,
                temperature=0.7
            ))
    
    def register_model(self, config: ModelConfig):
        """注册模型配置"""
        self.model_configs[config.model_name] = config
        if self.default_model is None:
            self.default_model = config.model_name
        logger.info(f"Registered model: {config.model_name} ({config.model_type})")
    
    async def load_model(self, model_name: str) -> bool:
        """加载模型"""
        if model_name not in self.model_configs:
            logger.error(f"Model {model_name} not found in configurations")
            return False
        
        config = self.model_configs[model_name]
        
        try:
            if config.model_type == "ollama":
                return await self._load_ollama_model(config)
            elif config.model_type == "transformers":
                return await self._load_transformers_model(config)
            elif config.model_type == "openai":
                return await self._load_openai_model(config)
            elif config.model_type == "chatgpt":
                return await self._load_chatgpt_model(config)
            elif config.model_type == "deepseek":
                return await self._load_deepseek_model(config)
            else:
                logger.error(f"Unsupported model type: {config.model_type}")
                return False
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {str(e)}")
            return False
    
    async def _load_ollama_model(self, config: ModelConfig) -> bool:
        """加载 Ollama 模型"""
        try:
            # 检查模型是否已安装
            models = ollama.list()
            model_names = [model['name'] for model in models['models']]
            
            if config.model_name not in model_names:
                logger.info(f"Pulling Ollama model: {config.model_name}")
                ollama.pull(config.model_name)
            
            # 测试模型
            response = ollama.generate(
                model=config.model_name,
                prompt="Hello",
                options={
                    'num_predict': 10,
                    'temperature': config.temperature
                }
            )
            
            self.models[config.model_name] = {
                'type': 'ollama',
                'config': config,
                'loaded': True
            }
            
            logger.info(f"Ollama model {config.model_name} loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load Ollama model {config.model_name}: {str(e)}")
            return False
    
    async def _load_transformers_model(self, config: ModelConfig) -> bool:
        """加载 Transformers 模型"""
        try:
            # 检查是否有GPU
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            # 加载分词器和模型
            tokenizer = AutoTokenizer.from_pretrained(config.model_name)
            model = AutoModelForCausalLM.from_pretrained(
                config.model_name,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map="auto" if device == "cuda" else None
            )
            
            # 创建pipeline
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                device=0 if device == "cuda" else -1,
                max_new_tokens=config.max_tokens,
                temperature=config.temperature,
                do_sample=True
            )
            
            self.models[config.model_name] = {
                'type': 'transformers',
                'config': config,
                'pipeline': pipe,
                'tokenizer': tokenizer,
                'model': model,
                'loaded': True
            }
            
            logger.info(f"Transformers model {config.model_name} loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load Transformers model {config.model_name}: {str(e)}")
            return False
    
    async def _load_openai_model(self, config: ModelConfig) -> bool:
        """加载 OpenAI 模型"""
        try:
            client = openai.OpenAI(
                api_key=config.api_key,
                base_url=config.api_base
            )
            
            # 测试连接
            response = client.chat.completions.create(
                model=config.model_name,
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=10,
                temperature=config.temperature
            )
            
            self.models[config.model_name] = {
                'type': 'openai',
                'config': config,
                'client': client,
                'loaded': True
            }
            
            logger.info(f"OpenAI model {config.model_name} loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load OpenAI model {config.model_name}: {str(e)}")
            return False
    
    async def _load_chatgpt_model(self, config: ModelConfig) -> bool:
        """加载 ChatGPT 模型"""
        if not chatgpt_client.is_available():
            logger.error("ChatGPT API client not available")
            return False
        
        try:
            # 测试连接
            test_success = await chatgpt_client.test_connection()
            if not test_success:
                logger.error(f"ChatGPT model {config.model_name} connection test failed")
                return False
            
            self.models[config.model_name] = {
                'type': 'chatgpt',
                'config': config,
                'client': chatgpt_client,
                'loaded': True
            }
            
            logger.info(f"ChatGPT model {config.model_name} loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load ChatGPT model {config.model_name}: {str(e)}")
            return False
    
    async def _load_deepseek_model(self, config: ModelConfig) -> bool:
        """加载 DeepSeek 模型"""
        if not deepseek_client.is_available():
            logger.error("DeepSeek API client not available")
            return False
        
        try:
            # 测试连接
            test_success = await deepseek_client.test_connection()
            if not test_success:
                logger.error(f"DeepSeek model {config.model_name} connection test failed")
                return False
            
            self.models[config.model_name] = {
                'type': 'deepseek',
                'config': config,
                'client': deepseek_client,
                'loaded': True
            }
            
            logger.info(f"DeepSeek model {config.model_name} loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load DeepSeek model {config.model_name}: {str(e)}")
            return False
    
    async def generate_response(self, 
                              prompt: str, 
                              model_name: Optional[str] = None,
                              **kwargs) -> Optional[str]:
        """生成响应"""
        model_name = model_name or self.default_model
        
        if not model_name or model_name not in self.models:
            # 尝试加载模型
            if model_name and await self.load_model(model_name):
                pass
            else:
                logger.error(f"Model {model_name} not available")
                return None
        
        model_info = self.models[model_name]
        config = model_info['config']
        
        try:
            if model_info['type'] == 'ollama':
                return await self._generate_ollama_response(prompt, model_info, **kwargs)
            elif model_info['type'] == 'transformers':
                return await self._generate_transformers_response(prompt, model_info, **kwargs)
            elif model_info['type'] == 'openai':
                return await self._generate_openai_response(prompt, model_info, **kwargs)
            elif model_info['type'] == 'chatgpt':
                return await self._generate_chatgpt_response(prompt, model_info, **kwargs)
            elif model_info['type'] == 'deepseek':
                return await self._generate_deepseek_response(prompt, model_info, **kwargs)
            else:
                logger.error(f"Unsupported model type: {model_info['type']}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to generate response with {model_name}: {str(e)}")
            return None
    
    async def _generate_ollama_response(self, prompt: str, model_info: Dict, **kwargs) -> str:
        """使用 Ollama 生成响应"""
        config = model_info['config']
        
        response = ollama.generate(
            model=config.model_name,
            prompt=prompt,
            options={
                'num_predict': kwargs.get('max_tokens', config.max_tokens),
                'temperature': kwargs.get('temperature', config.temperature),
                **kwargs
            }
        )
        
        return response['response']
    
    async def _generate_transformers_response(self, prompt: str, model_info: Dict, **kwargs) -> str:
        """使用 Transformers 生成响应"""
        pipeline = model_info['pipeline']
        config = model_info['config']
        
        # 在新线程中运行以避免阻塞
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: pipeline(
                prompt,
                max_new_tokens=kwargs.get('max_tokens', config.max_tokens),
                temperature=kwargs.get('temperature', config.temperature),
                do_sample=True,
                pad_token_id=pipeline.tokenizer.eos_token_id
            )
        )
        
        # 提取生成的文本
        generated_text = response[0]['generated_text']
        # 移除原始prompt
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):].strip()
        
        return generated_text
    
    async def _generate_openai_response(self, prompt: str, model_info: Dict, **kwargs) -> str:
        """使用 OpenAI 生成响应"""
        client = model_info['client']
        config = model_info['config']
        
        response = client.chat.completions.create(
            model=config.model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=kwargs.get('max_tokens', config.max_tokens),
            temperature=kwargs.get('temperature', config.temperature)
        )
        
        return response.choices[0].message.content
    
    async def _generate_chatgpt_response(self, prompt: str, model_info: Dict, **kwargs) -> str:
        """使用 ChatGPT 生成响应"""
        client = model_info['client']
        config = model_info['config']
        
        # 准备消息
        messages = [ChatMessage(role="user", content=prompt)]
        
        # 调用 ChatGPT API
        response = await client.chat_completion(
            messages=messages,
            model=config.model_name.replace("chatgpt-", "gpt-"),  # 转换模型名称
            temperature=kwargs.get('temperature', config.temperature),
            max_tokens=kwargs.get('max_tokens', config.max_tokens)
        )
        
        return response.content
    
    async def _generate_deepseek_response(self, prompt: str, model_info: Dict, **kwargs) -> str:
        """使用 DeepSeek 生成响应"""
        client = model_info['client']
        config = model_info['config']
        
        # 准备消息
        messages = [DeepSeekMessage(role="user", content=prompt)]
        
        # 调用 DeepSeek API
        response = await client.chat_completion(
            messages=messages,
            model=config.model_name,
            temperature=kwargs.get('temperature', config.temperature),
            max_tokens=kwargs.get('max_tokens', config.max_tokens)
        )
        
        return response.content
    
    def list_models(self) -> List[Dict[str, Any]]:
        """列出所有可用模型"""
        models = []
        for name, config in self.model_configs.items():
            models.append({
                'name': name,
                'type': config.model_type,
                'loaded': name in self.models and self.models[name].get('loaded', False),
                'is_default': name == self.default_model
            })
        return models
    
    def set_default_model(self, model_name: str) -> bool:
        """设置默认模型"""
        if model_name in self.model_configs:
            self.default_model = model_name
            logger.info(f"Default model set to: {model_name}")
            return True
        return False
    
    def unload_model(self, model_name: str) -> bool:
        """卸载模型"""
        if model_name in self.models:
            del self.models[model_name]
            logger.info(f"Model {model_name} unloaded")
            return True
        return False
    
    async def multi_turn_chat(self, 
                             conversation_history: List[Dict[str, str]], 
                             new_message: str,
                             model_name: Optional[str] = None,
                             **kwargs) -> Optional[str]:
        """
        多轮对话
        
        Args:
            conversation_history: 对话历史 [{"role": "user/assistant", "content": "..."}]
            new_message: 新消息
            model_name: 模型名称
            **kwargs: 其他参数
        
        Returns:
            str: AI 回复
        """
        model_name = model_name or self.default_model
        
        if not model_name or model_name not in self.models:
            if model_name and await self.load_model(model_name):
                pass
            else:
                logger.error(f"Model {model_name} not available")
                return None
        
        model_info = self.models[model_name]
        config = model_info['config']
        
        try:
            if model_info['type'] == 'chatgpt':
                # 转换对话历史为 ChatMessage 格式
                messages = [ChatMessage(role=msg["role"], content=msg["content"]) 
                           for msg in conversation_history]
                messages.append(ChatMessage(role="user", content=new_message))
                
                client = model_info['client']
                response = await client.multi_turn_chat(messages[:-1], new_message)
                return response.content
                
            elif model_info['type'] == 'deepseek':
                # 转换对话历史为 DeepSeekMessage 格式
                messages = [DeepSeekMessage(role=msg["role"], content=msg["content"]) 
                           for msg in conversation_history]
                messages.append(DeepSeekMessage(role="user", content=new_message))
                
                client = model_info['client']
                response = await client.multi_turn_chat(messages[:-1], new_message)
                return response.content
                
            else:
                # 对于其他模型类型，将对话历史合并为单个 prompt
                prompt_parts = []
                for msg in conversation_history:
                    role = "Human" if msg["role"] == "user" else "Assistant"
                    prompt_parts.append(f"{role}: {msg['content']}")
                prompt_parts.append(f"Human: {new_message}")
                prompt_parts.append("Assistant:")
                
                full_prompt = "\n".join(prompt_parts)
                return await self.generate_response(full_prompt, model_name, **kwargs)
                
        except Exception as e:
            logger.error(f"Multi-turn chat failed with {model_name}: {str(e)}")
            return None
    
    async def stream_chat(self, 
                         messages: List[Dict[str, str]], 
                         model_name: Optional[str] = None,
                         **kwargs):
        """
        流式聊天
        
        Args:
            messages: 消息列表 [{"role": "user/assistant", "content": "..."}]
            model_name: 模型名称
            **kwargs: 其他参数
        
        Yields:
            str: 流式响应内容
        """
        model_name = model_name or self.default_model
        
        if not model_name or model_name not in self.models:
            if model_name and await self.load_model(model_name):
                pass
            else:
                logger.error(f"Model {model_name} not available")
                return
        
        model_info = self.models[model_name]
        
        try:
            if model_info['type'] == 'chatgpt':
                # 转换消息格式
                chat_messages = [ChatMessage(role=msg["role"], content=msg["content"]) 
                               for msg in messages]
                
                client = model_info['client']
                async for chunk in client.stream_chat(chat_messages, **kwargs):
                    yield chunk
                    
            elif model_info['type'] == 'deepseek':
                # 转换消息格式
                deepseek_messages = [DeepSeekMessage(role=msg["role"], content=msg["content"]) 
                                   for msg in messages]
                
                client = model_info['client']
                async for chunk in client.stream_chat(deepseek_messages, **kwargs):
                    yield chunk
                    
            else:
                # 对于不支持流式的模型，返回完整响应
                prompt_parts = []
                for msg in messages:
                    role = "Human" if msg["role"] == "user" else "Assistant"
                    prompt_parts.append(f"{role}: {msg['content']}")
                prompt_parts.append("Assistant:")
                
                full_prompt = "\n".join(prompt_parts)
                response = await self.generate_response(full_prompt, model_name, **kwargs)
                if response:
                    yield response
                    
        except Exception as e:
            logger.error(f"Stream chat failed with {model_name}: {str(e)}")
    
    def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """获取模型信息"""
        if model_name in self.model_configs:
            config = self.model_configs[model_name]
            is_loaded = model_name in self.models and self.models[model_name].get('loaded', False)
            
            return {
                'name': model_name,
                'type': config.model_type,
                'loaded': is_loaded,
                'is_default': model_name == self.default_model,
                'max_tokens': config.max_tokens,
                'temperature': config.temperature,
                'supports_streaming': config.model_type in ['chatgpt', 'deepseek'],
                'supports_multi_turn': True
            }
        return None
    
    def get_available_providers(self) -> List[str]:
        """获取可用的提供商列表"""
        providers = set()
        for config in self.model_configs.values():
            providers.add(config.model_type)
        return list(providers)


# 全局实例
_local_model_service = None


def get_local_model_service() -> LocalModelService:
    """获取本地模型服务实例"""
    global _local_model_service
    if _local_model_service is None:
        _local_model_service = LocalModelService()
    return _local_model_service
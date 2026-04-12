#!/usr/bin/env python3
"""
多提供者 LLM 服务演示脚本
测试 ChatGPT、DeepSeek 和本地模型的集成
"""

import asyncio
import logging
from backend.services.langchain_inference_service import LangChainInferenceService

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_multi_provider():
    """测试多提供者会话功能"""
    
    # 初始化服务
    service = LangChainInferenceService()
    
    # 测试消息
    test_message = "你好，请简单介绍一下你自己。"
    
    # 测试不同的提供者
    providers = ["local", "chatgpt", "deepseek"]
    
    for provider in providers:
        logger.info(f"\n{'='*50}")
        logger.info(f"测试提供者: {provider}")
        logger.info(f"{'='*50}")
        
        try:
            # 创建会话
            session_id = await service.create_session(
                user_id="test_user",
                agent_id="test_agent",
                memory_type="buffer",
                max_token_limit=2000
            )
            logger.info(f"创建会话: {session_id}")
            
            # 发送消息
            response = await service.chat(
                session_id=session_id,
                message=test_message,
                provider=provider,
                model_name="llama2" if provider == "local" else None
            )
            
            logger.info(f"用户消息: {test_message}")
            logger.info(f"AI 回复: {response}")
            
        except Exception as e:
            logger.error(f"提供者 {provider} 测试失败: {str(e)}")
            import traceback
            traceback.print_exc()
    
    logger.info(f"\n{'='*50}")
    logger.info("多提供者测试完成")
    logger.info(f"{'='*50}")

if __name__ == "__main__":
    asyncio.run(test_multi_provider())
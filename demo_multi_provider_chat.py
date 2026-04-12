"""多提供者会话演示脚本

演示训练助手智能体的多提供者功能，包括：
- 本地模型会话
- ChatGPT 会话
- DeepSeek 会话
- 提供者切换
- 流式对话
"""

import asyncio
import logging
import sys
import os

# 添加项目根目录到 Python 路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backend.modules.agent.training_assistant_agent import TrainingAssistantAgent
from backend.services.langchain_inference_service import get_langchain_inference_service
from backend.schemas.agent import Agent

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MultiProviderChatDemo:
    """多提供者会话演示"""
    
    def __init__(self):
        self.training_agent = None
        self.langchain_service = None
        self.current_session_id = "demo_session_001"
        
    async def initialize_services(self):
        """初始化服务"""
        try:
            logger.info("初始化服务...")
            
            # 创建演示用的智能体模型
            demo_agent_model = Agent(
                agent_id="demo_agent",
                user_id="demo_user",
                name="演示训练助手",
                description="用于演示多提供者功能的训练助手",
                capabilities=["training", "chat", "multi_provider"],
                active=True
            )
            
            # 初始化训练助手
            self.training_agent = TrainingAssistantAgent(demo_agent_model)
            
            # 获取 LangChain 推理服务
            self.langchain_service = get_langchain_inference_service()
            
            logger.info("服务初始化完成")
            return True
            
        except Exception as e:
            logger.error(f"服务初始化失败: {str(e)}")
            return False
    
    async def show_available_providers(self):
        """显示可用的提供者"""
        logger.info("\n" + "="*50)
        logger.info("可用的推理提供者")
        logger.info("="*50)
        
        try:
            # 获取训练助手的可用提供者
            providers = self.training_agent.get_available_providers()
            logger.info(f"训练助手支持的提供者: {providers}")
            
            # 获取 LangChain 服务的可用提供者
            langchain_providers = self.langchain_service.get_available_providers()
            logger.info("LangChain 服务提供者状态:")
            for provider in langchain_providers:
                status = "✓ 可用" if provider['available'] else "✗ 不可用"
                logger.info(f"  - {provider['display_name']} ({provider['name']}): {status}")
                if provider['models']:
                    logger.info(f"    支持的模型: {', '.join(provider['models'])}")
                    
        except Exception as e:
            logger.error(f"获取推理服务提供者失败: {str(e)}")
    
    async def create_session(self):
        """创建会话"""
        try:
            session_id = await self.langchain_service.create_session(
                user_id="demo_user",
                agent_id="demo_agent",
                memory_type="buffer",
                max_token_limit=2000
            )
            self.current_session_id = session_id
            logger.info(f"创建会话成功: {self.current_session_id}")
            return session_id
        except Exception as e:
            logger.error(f"创建会话失败: {str(e)}")
            return None
    
    async def chat_with_provider(self, provider: str, message: str):
        """与指定提供者对话"""
        logger.info(f"\n--- 使用 {provider} 进行对话 ---")
        logger.info(f"用户: {message}")
        
        try:
            # 使用训练助手的 process 方法
            input_data = {
                "user_request": message,
                "user_id": "demo_user",
                "session_id": self.current_session_id,
                "use_inference": True,
                "provider": provider  # 直接在输入数据中指定提供者
            }
            
            context = {
                "provider": provider,
                "demo": True
            }
            
            result = await self.training_agent.process(input_data, context)
            
            if result.get("status") == "success":
                response = result.get("response", "无响应")
                logger.info(f"{provider}: {response}")
            else:
                logger.error(f"{provider} 对话失败: {result.get('error', '未知错误')}")
                
        except Exception as e:
            logger.error(f"与 {provider} 对话失败: {str(e)}")
    
    async def multi_turn_conversation(self, provider: str):
        """多轮对话演示"""
        logger.info(f"\n--- {provider} 多轮对话演示 ---")
        
        messages = [
            "你好，我想了解模型训练的基本流程",
            "训练过程中如何监控模型性能？",
            "训练完成后如何评估模型效果？"
        ]
        
        for i, message in enumerate(messages, 1):
            logger.info(f"\n第 {i} 轮对话:")
            await self.chat_with_provider(provider, message)
            await asyncio.sleep(1)  # 短暂延迟
    
    async def compare_providers(self, message: str):
        """比较不同提供者的响应"""
        logger.info(f"\n--- 提供者响应比较 ---")
        logger.info(f"测试问题: {message}")
        
        providers = ["local", "chatgpt", "deepseek"]
        
        for provider in providers:
            await self.chat_with_provider(provider, message)
            await asyncio.sleep(1)
    
    async def switch_provider_demo(self):
        """提供者切换演示"""
        logger.info("\n--- 提供者切换演示 ---")
        
        # 模拟在对话中切换提供者
        conversations = [
            ("local", "请介绍一下深度学习的基本概念"),
            ("chatgpt", "继续上面的话题，详细说明神经网络的工作原理"),
            ("deepseek", "基于前面的讨论，如何选择合适的优化器？")
        ]
        
        for provider, message in conversations:
            logger.info(f"\n切换到 {provider} 提供者:")
            await self.chat_with_provider(provider, message)
            await asyncio.sleep(1)
    
    async def stream_chat_demo(self, provider: str, message: str):
        """流式对话演示"""
        logger.info(f"\n--- {provider} 流式对话演示 ---")
        logger.info(f"用户: {message}")
        
        try:
            # 使用训练助手的流式对话功能
            print(f"{provider} (流式): ", end="", flush=True)
            
            async for chunk in self.training_agent.stream_chat(
                session_id=self.current_session_id,
                message=message,
                user_id="demo_user",
                provider=provider
            ):
                if chunk:
                    print(chunk, end="", flush=True)
            
            print()  # 换行
            
        except Exception as e:
            logger.error(f"{provider} 流式对话失败: {str(e)}")
    
    async def show_provider_status(self):
        """显示提供者状态"""
        logger.info("\n--- 提供者详细状态 ---")
        
        providers = ["local", "chatgpt", "deepseek"]
        
        for provider in providers:
            try:
                status = self.langchain_service.get_provider_status(provider)
                logger.info(f"\n{provider.upper()} 状态:")
                logger.info(f"  可用性: {'✓' if status['available'] else '✗'}")
                logger.info(f"  支持的模型: {status.get('models', [])}")
                if 'default_model' in status:
                    logger.info(f"  默认模型: {status['default_model']}")
                    
            except Exception as e:
                logger.error(f"获取 {provider} 状态失败: {str(e)}")
    
    async def test_direct_langchain_chat(self, provider: str, message: str):
        """直接测试 LangChain 服务的对话功能"""
        logger.info(f"\n--- 直接测试 {provider} LangChain 对话 ---")
        logger.info(f"用户: {message}")
        
        try:
            response = await self.langchain_service.chat(
                session_id=self.current_session_id,
                message=message,
                provider=provider
            )
            
            if response:
                logger.info(f"{provider} (直接): {response}")
            else:
                logger.error(f"{provider} 直接对话失败: 无响应")
                
        except Exception as e:
            logger.error(f"{provider} 直接对话失败: {str(e)}")
    
    async def run_demo(self):
        """运行完整演示"""
        logger.info("开始多提供者会话演示")
        
        # 初始化服务
        if not await self.initialize_services():
            logger.error("服务初始化失败，演示退出")
            return
        
        # 创建会话
        if not await self.create_session():
            logger.error("会话创建失败，演示退出")
            return
        
        # 显示可用提供者
        await self.show_available_providers()
        
        # 显示提供者详细状态
        await self.show_provider_status()
        
        # 直接测试 LangChain 服务
        test_message = "你好，我想了解如何开始一个机器学习项目"
        await self.test_direct_langchain_chat("local", test_message)
        
        # 基本对话测试
        await self.chat_with_provider("local", test_message)
        
        # 比较不同提供者
        await self.compare_providers("什么是过拟合，如何避免？")
        
        # 多轮对话演示
        await self.multi_turn_conversation("local")
        
        # 提供者切换演示
        await self.switch_provider_demo()
        
        # 流式对话演示
        await self.stream_chat_demo("local", "请详细解释梯度下降算法的工作原理")
        
        logger.info("\n演示完成！")


async def main():
    """主函数"""
    demo = MultiProviderChatDemo()
    await demo.run_demo()


if __name__ == "__main__":
    asyncio.run(main())
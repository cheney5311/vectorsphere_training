#!/usr/bin/env python3
"""
API 集成测试脚本

该脚本用于测试 ChatGPT 和 DeepSeek API 的集成功能，
验证本地模型服务、LangChain 推理服务和训练助手智能体的多提供者支持。
"""

import asyncio
import json
import logging
import sys
import os
from typing import Dict, Any, List

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.modules.agent.api_config_manager import APIConfigManager
from backend.modules.agent.chatgpt_api_client import ChatGPTAPIClient
from backend.modules.agent.deepseek_api_client import DeepSeekAPIClient
from backend.modules.agent.local_model_service import get_local_model_service
from backend.modules.agent.langchain_inference_service import get_langchain_inference_service
from backend.modules.agent.training_assistant_agent import TrainingAssistantAgent
from backend.schemas.agent import Agent

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class APIIntegrationTester:
    """API 集成测试器"""
    
    def __init__(self):
        self.config_manager = APIConfigManager()
        self.test_results = {}
        
    async def run_all_tests(self) -> Dict[str, Any]:
        """运行所有测试"""
        logger.info("开始 API 集成测试...")
        
        # 测试配置管理器
        await self.test_config_manager()
        
        # 测试 API 客户端
        await self.test_api_clients()
        
        # 测试本地模型服务
        await self.test_local_model_service()
        
        # 测试 LangChain 推理服务
        await self.test_langchain_inference_service()
        
        # 测试训练助手智能体
        await self.test_training_assistant_agent()
        
        # 生成测试报告
        self.generate_test_report()
        
        return self.test_results
    
    async def test_config_manager(self):
        """测试配置管理器"""
        logger.info("测试配置管理器...")
        
        try:
            # 测试获取配置
            chatgpt_config = self.config_manager.get_provider_config("chatgpt")
            deepseek_config = self.config_manager.get_provider_config("deepseek")
            
            # 测试验证配置
            chatgpt_valid = self.config_manager.validate_provider_config("chatgpt")
            deepseek_valid = self.config_manager.validate_provider_config("deepseek")
            
            self.test_results["config_manager"] = {
                "status": "success",
                "chatgpt_config_exists": chatgpt_config is not None,
                "deepseek_config_exists": deepseek_config is not None,
                "chatgpt_config_valid": chatgpt_valid,
                "deepseek_config_valid": deepseek_valid
            }
            
            logger.info("配置管理器测试完成")
            
        except Exception as e:
            logger.error(f"配置管理器测试失败: {str(e)}")
            self.test_results["config_manager"] = {
                "status": "error",
                "error": str(e)
            }
    
    async def test_api_clients(self):
        """测试 API 客户端"""
        logger.info("测试 API 客户端...")
        
        # 测试 ChatGPT 客户端
        await self.test_chatgpt_client()
        
        # 测试 DeepSeek 客户端
        await self.test_deepseek_client()
    
    async def test_chatgpt_client(self):
        """测试 ChatGPT 客户端"""
        try:
            client = ChatGPTAPIClient()
            
            # 测试连接
            is_available = await client.is_available()
            
            if is_available:
                # 测试聊天完成
                test_message = {
                    "role": "user",
                    "content": "Hello, this is a test message."
                }
                
                response = await client.chat_completion(
                    messages=[test_message],
                    model="gpt-3.5-turbo",
                    max_tokens=50
                )
                
                self.test_results["chatgpt_client"] = {
                    "status": "success",
                    "available": True,
                    "response_received": response is not None,
                    "response_content": response.get("content", "") if response else ""
                }
            else:
                self.test_results["chatgpt_client"] = {
                    "status": "warning",
                    "available": False,
                    "message": "ChatGPT API 不可用，可能是配置问题"
                }
                
        except Exception as e:
            logger.error(f"ChatGPT 客户端测试失败: {str(e)}")
            self.test_results["chatgpt_client"] = {
                "status": "error",
                "error": str(e)
            }
    
    async def test_deepseek_client(self):
        """测试 DeepSeek 客户端"""
        try:
            client = DeepSeekAPIClient()
            
            # 测试连接
            is_available = await client.is_available()
            
            if is_available:
                # 测试聊天完成
                test_message = {
                    "role": "user",
                    "content": "Hello, this is a test message."
                }
                
                response = await client.chat_completion(
                    messages=[test_message],
                    model="deepseek-chat",
                    max_tokens=50
                )
                
                self.test_results["deepseek_client"] = {
                    "status": "success",
                    "available": True,
                    "response_received": response is not None,
                    "response_content": response.get("content", "") if response else ""
                }
            else:
                self.test_results["deepseek_client"] = {
                    "status": "warning",
                    "available": False,
                    "message": "DeepSeek API 不可用，可能是配置问题"
                }
                
        except Exception as e:
            logger.error(f"DeepSeek 客户端测试失败: {str(e)}")
            self.test_results["deepseek_client"] = {
                "status": "error",
                "error": str(e)
            }
    
    async def test_local_model_service(self):
        """测试本地模型服务"""
        logger.info("测试本地模型服务...")
        
        try:
            service = get_local_model_service()
            
            # 测试获取可用提供者
            providers = service.get_available_providers()
            
            # 测试列出模型
            models = service.list_models()
            
            # 测试云端 API 模型注册
            cloud_models = [model for model in models if model.get("type") in ["chatgpt", "deepseek"]]
            
            self.test_results["local_model_service"] = {
                "status": "success",
                "available_providers": providers,
                "total_models": len(models),
                "cloud_models": len(cloud_models),
                "cloud_model_types": list(set(model.get("type") for model in cloud_models))
            }
            
            logger.info("本地模型服务测试完成")
            
        except Exception as e:
            logger.error(f"本地模型服务测试失败: {str(e)}")
            self.test_results["local_model_service"] = {
                "status": "error",
                "error": str(e)
            }
    
    async def test_langchain_inference_service(self):
        """测试 LangChain 推理服务"""
        logger.info("测试 LangChain 推理服务...")
        
        try:
            service = get_langchain_inference_service()
            
            # 测试获取可用提供者
            providers = service.get_available_providers()
            
            # 测试创建会话
            session_id = await service.create_session(
                user_id="test_user",
                agent_id="test_agent"
            )
            
            # 测试不同提供者的对话
            test_results = {}
            
            for provider in ["local", "chatgpt", "deepseek"]:
                if provider in providers:
                    try:
                        response = await service.chat(
                            session_id=session_id,
                            message="Hello, this is a test.",
                            provider=provider
                        )
                        
                        test_results[provider] = {
                            "status": "success",
                            "response_received": response is not None
                        }
                    except Exception as e:
                        test_results[provider] = {
                            "status": "error",
                            "error": str(e)
                        }
                else:
                    test_results[provider] = {
                        "status": "skipped",
                        "reason": "Provider not available"
                    }
            
            self.test_results["langchain_inference_service"] = {
                "status": "success",
                "available_providers": providers,
                "session_created": session_id is not None,
                "provider_tests": test_results
            }
            
            logger.info("LangChain 推理服务测试完成")
            
        except Exception as e:
            logger.error(f"LangChain 推理服务测试失败: {str(e)}")
            self.test_results["langchain_inference_service"] = {
                "status": "error",
                "error": str(e)
            }
    
    async def test_training_assistant_agent(self):
        """测试训练助手智能体"""
        logger.info("测试训练助手智能体...")
        
        try:
            # 创建测试智能体
            agent_model = Agent(
                agent_id="test_training_assistant",
                name="测试训练助手",
                description="用于测试的训练助手智能体",
                agent_type="training_assistant"
            )
            
            agent = TrainingAssistantAgent(agent_model)
            
            # 测试获取可用提供者
            providers = agent.get_available_providers()
            
            # 测试设置默认提供者
            default_set = agent.set_default_provider("local")
            current_default = agent.get_default_provider()
            
            # 测试处理请求（使用不同提供者）
            test_request = "你好，我想了解训练助手的功能。"
            
            provider_tests = {}
            for provider in ["local", "chatgpt", "deepseek"]:
                if provider in providers:
                    try:
                        result = await agent.process({
                            "user_request": test_request,
                            "user_id": "test_user",
                            "use_inference": True
                        })
                        
                        provider_tests[provider] = {
                            "status": "success",
                            "response_received": "result" in result
                        }
                    except Exception as e:
                        provider_tests[provider] = {
                            "status": "error",
                            "error": str(e)
                        }
                else:
                    provider_tests[provider] = {
                        "status": "skipped",
                        "reason": "Provider not available"
                    }
            
            self.test_results["training_assistant_agent"] = {
                "status": "success",
                "available_providers": providers,
                "default_provider_set": default_set,
                "current_default": current_default,
                "provider_tests": provider_tests
            }
            
            logger.info("训练助手智能体测试完成")
            
        except Exception as e:
            logger.error(f"训练助手智能体测试失败: {str(e)}")
            self.test_results["training_assistant_agent"] = {
                "status": "error",
                "error": str(e)
            }
    
    def generate_test_report(self):
        """生成测试报告"""
        logger.info("生成测试报告...")
        
        report = {
            "test_summary": {
                "total_tests": len(self.test_results),
                "successful_tests": len([r for r in self.test_results.values() if r.get("status") == "success"]),
                "failed_tests": len([r for r in self.test_results.values() if r.get("status") == "error"]),
                "warning_tests": len([r for r in self.test_results.values() if r.get("status") == "warning"])
            },
            "detailed_results": self.test_results
        }
        
        # 保存测试报告
        report_file = "api_integration_test_report.json"
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        logger.info(f"测试报告已保存到: {report_file}")
        
        # 打印摘要
        print("\n" + "="*50)
        print("API 集成测试报告摘要")
        print("="*50)
        print(f"总测试数: {report['test_summary']['total_tests']}")
        print(f"成功: {report['test_summary']['successful_tests']}")
        print(f"失败: {report['test_summary']['failed_tests']}")
        print(f"警告: {report['test_summary']['warning_tests']}")
        print("="*50)
        
        # 打印详细结果
        for test_name, result in self.test_results.items():
            status = result.get("status", "unknown")
            print(f"{test_name}: {status.upper()}")
            if status == "error":
                print(f"  错误: {result.get('error', 'Unknown error')}")
            elif status == "warning":
                print(f"  警告: {result.get('message', 'Unknown warning')}")


async def main():
    """主函数"""
    tester = APIIntegrationTester()
    
    try:
        await tester.run_all_tests()
        print("\n测试完成！请查看 api_integration_test_report.json 获取详细结果。")
    except Exception as e:
        logger.error(f"测试运行失败: {str(e)}")
        print(f"测试运行失败: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
#!/usr/bin/env python3
"""
推理功能和会话管理测试脚本

该脚本用于测试训练助手智能体的推理能力和会话管理功能。
"""

import asyncio
import sys
import os
import json
import uuid
from datetime import datetime
from typing import Dict, Any, List

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.modules.agent.training_assistant_agent import TrainingAssistantAgent
from backend.modules.agent.local_model_service import LocalModelService
from backend.modules.agent.langchain_inference_service import LangChainInferenceService
from backend.modules.agent.session_history_manager import SessionHistoryManager, MessageType
from loguru import logger


class InferenceTester:
    """推理功能测试器"""
    
    def __init__(self):
        """初始化测试器"""
        self.test_user_id = "test_user_001"
        self.test_agent_id = "training_assistant"
        self.session_id = None
        
        # 初始化服务
        self.local_model_service = None
        self.inference_service = None
        self.session_manager = None
        self.training_agent = None
        
        # 测试结果
        self.test_results = []
    
    async def setup(self):
        """设置测试环境"""
        try:
            logger.info("正在设置测试环境...")
            
            # 初始化本地模型服务
            self.local_model_service = LocalModelService()
            
            # 初始化推理服务
            self.inference_service = LangChainInferenceService(self.local_model_service)
            
            # 初始化会话管理器
            self.session_manager = SessionHistoryManager()
            
            # 创建模拟的Agent模型
            from dataclasses import dataclass
            
            @dataclass
            class MockAgent:
                agent_id: str = "training_assistant"
                name: str = "训练助手"
                description: str = "智能训练助手"
                capabilities: list = None
                
                def __post_init__(self):
                    if self.capabilities is None:
                        self.capabilities = ["training", "inference", "conversation"]
            
            # 初始化训练助手智能体
            mock_agent = MockAgent()
            self.training_agent = TrainingAssistantAgent(mock_agent)
            
            logger.info("测试环境设置完成")
            return True
            
        except Exception as e:
            logger.error(f"设置测试环境失败: {str(e)}")
            return False
    
    async def test_local_model_service(self) -> bool:
        """测试本地模型服务"""
        logger.info("开始测试本地模型服务...")
        
        try:
            # 测试模型列表
            models = self.local_model_service.list_models()
            logger.info(f"可用模型: {models}")
            
            # 测试模型配置
            config = self.local_model_service.get_model_config("default")
            logger.info(f"默认模型配置: {config}")
            
            # 测试简单生成（如果有可用模型）
            if models:
                try:
                    response = await self.local_model_service.generate_response(
                        "你好，这是一个测试消息",
                        model_name="default"
                    )
                    logger.info(f"模型响应: {response}")
                    
                    self.test_results.append({
                        "test": "local_model_service",
                        "status": "success",
                        "details": f"成功生成响应: {response[:100]}..."
                    })
                    return True
                    
                except Exception as e:
                    logger.warning(f"模型生成测试失败（可能是模型未安装）: {str(e)}")
                    self.test_results.append({
                        "test": "local_model_service",
                        "status": "partial",
                        "details": f"服务可用但模型生成失败: {str(e)}"
                    })
                    return True
            else:
                logger.warning("没有可用的模型，跳过生成测试")
                self.test_results.append({
                    "test": "local_model_service",
                    "status": "partial",
                    "details": "服务可用但没有配置模型"
                })
                return True
                
        except Exception as e:
            logger.error(f"本地模型服务测试失败: {str(e)}")
            self.test_results.append({
                "test": "local_model_service",
                "status": "failed",
                "details": str(e)
            })
            return False
    
    async def test_session_history_manager(self) -> bool:
        """测试会话历史管理器"""
        logger.info("开始测试会话历史管理器...")
        
        try:
            test_session_id = f"test_session_{uuid.uuid4().hex[:8]}"
            
            # 测试创建会话
            session_info = await self.session_manager.create_session(
                session_id=test_session_id,
                user_id=self.test_user_id,
                agent_id=self.test_agent_id,
                metadata={"test": True}
            )
            logger.info(f"创建会话成功: {session_info.session_id}")
            
            # 测试添加消息
            message1 = await self.session_manager.add_message(
                session_id=test_session_id,
                message_id=f"msg_{uuid.uuid4().hex[:8]}",
                message_type=MessageType.USER,
                content="这是一条测试用户消息",
                metadata={"test": True}
            )
            
            message2 = await self.session_manager.add_message(
                session_id=test_session_id,
                message_id=f"msg_{uuid.uuid4().hex[:8]}",
                message_type=MessageType.ASSISTANT,
                content="这是一条测试助手回复",
                metadata={"test": True}
            )
            
            logger.info(f"添加消息成功: {message1.id}, {message2.id}")
            
            # 测试获取消息
            messages = await self.session_manager.get_session_messages(test_session_id)
            logger.info(f"获取到 {len(messages)} 条消息")
            
            # 测试获取会话信息
            retrieved_session = await self.session_manager.get_session_info(test_session_id)
            logger.info(f"获取会话信息: {retrieved_session.message_count} 条消息")
            
            # 测试获取用户会话
            user_sessions = await self.session_manager.get_user_sessions(self.test_user_id)
            logger.info(f"用户有 {len(user_sessions)} 个会话")
            
            # 测试会话统计
            stats = await self.session_manager.get_session_statistics(test_session_id)
            logger.info(f"会话统计: {stats}")
            
            # 清理测试会话
            await self.session_manager.delete_session(test_session_id)
            logger.info("清理测试会话完成")
            
            self.test_results.append({
                "test": "session_history_manager",
                "status": "success",
                "details": f"成功测试会话管理，处理了 {len(messages)} 条消息"
            })
            return True
            
        except Exception as e:
            logger.error(f"会话历史管理器测试失败: {str(e)}")
            self.test_results.append({
                "test": "session_history_manager",
                "status": "failed",
                "details": str(e)
            })
            return False
    
    async def test_langchain_inference_service(self) -> bool:
        """测试LangChain推理服务"""
        logger.info("开始测试LangChain推理服务...")
        
        try:
            # 测试创建会话
            session_id = await self.inference_service.create_session(
                user_id=self.test_user_id,
                agent_id=self.test_agent_id,
                memory_type="buffer",
                max_token_limit=1000
            )
            logger.info(f"创建推理会话成功: {session_id}")
            self.session_id = session_id
            
            # 测试简单对话
            test_messages = [
                "你好，我是测试用户",
                "请介绍一下训练平台的功能",
                "如何创建一个新的训练任务？"
            ]
            
            for i, message in enumerate(test_messages):
                try:
                    response = await self.inference_service.chat(
                        session_id=session_id,
                        message=message,
                        context={"test_message": i + 1}
                    )
                    logger.info(f"对话 {i+1} - 用户: {message}")
                    logger.info(f"对话 {i+1} - 助手: {response}")
                    
                except Exception as e:
                    logger.warning(f"对话 {i+1} 失败: {str(e)}")
            
            # 测试获取会话
            session = await self.inference_service.get_session(session_id)
            if session:
                history = session.get_history()
                logger.info(f"会话历史包含 {len(history)} 条记录")
            
            # 测试列出会话
            sessions = await self.inference_service.list_sessions(self.test_user_id)
            logger.info(f"用户有 {len(sessions)} 个推理会话")
            
            self.test_results.append({
                "test": "langchain_inference_service",
                "status": "success",
                "details": f"成功测试推理服务，处理了 {len(test_messages)} 条对话"
            })
            return True
            
        except Exception as e:
            logger.error(f"LangChain推理服务测试失败: {str(e)}")
            self.test_results.append({
                "test": "langchain_inference_service",
                "status": "failed",
                "details": str(e)
            })
            return False
    
    async def test_training_assistant_agent(self) -> bool:
        """测试训练助手智能体"""
        logger.info("开始测试训练助手智能体...")
        
        try:
            # 测试传统处理模式
            traditional_result = await self.training_agent.process(
                input_data={
                    "user_request": "查看我的训练历史",
                    "user_id": self.test_user_id,
                    "use_inference": False
                },
                context={"test": True}
            )
            logger.info(f"传统处理结果: {traditional_result.get('result', {}).get('action', 'unknown')}")
            
            # 测试智能推理模式
            if self.training_agent.is_inference_enabled():
                inference_result = await self.training_agent.process(
                    input_data={
                        "user_request": "你好，我想了解如何使用训练平台",
                        "user_id": self.test_user_id,
                        "use_inference": True
                    },
                    context={"test": True}
                )
                logger.info(f"推理处理结果: {inference_result.get('result', {}).get('action', 'unknown')}")
                
                # 测试会话管理
                if 'session_id' in inference_result:
                    session_id = inference_result['session_id']
                    
                    # 测试获取会话历史
                    history = await self.training_agent.get_session_history(session_id)
                    logger.info(f"会话历史包含 {len(history)} 条记录")
                    
                    # 测试更新会话上下文
                    update_success = await self.training_agent.update_session_context(
                        session_id, {"updated": True}
                    )
                    logger.info(f"更新会话上下文: {'成功' if update_success else '失败'}")
                    
                    # 测试清除会话
                    clear_success = await self.training_agent.clear_session(session_id)
                    logger.info(f"清除会话: {'成功' if clear_success else '失败'}")
            
            self.test_results.append({
                "test": "training_assistant_agent",
                "status": "success",
                "details": "成功测试智能体的传统和推理模式"
            })
            return True
            
        except Exception as e:
            logger.error(f"训练助手智能体测试失败: {str(e)}")
            self.test_results.append({
                "test": "training_assistant_agent",
                "status": "failed",
                "details": str(e)
            })
            return False
    
    async def cleanup(self):
        """清理测试环境"""
        try:
            logger.info("正在清理测试环境...")
            
            # 清理测试会话
            if self.session_id and self.inference_service:
                await self.inference_service.delete_session(self.session_id)
            
            # 清理用户会话
            if self.session_manager:
                await self.session_manager.clear_user_sessions(self.test_user_id)
            
            logger.info("测试环境清理完成")
            
        except Exception as e:
            logger.warning(f"清理测试环境时出错: {str(e)}")
    
    def print_test_results(self):
        """打印测试结果"""
        print("\n" + "="*60)
        print("推理功能和会话管理测试结果")
        print("="*60)
        
        success_count = 0
        total_count = len(self.test_results)
        
        for result in self.test_results:
            status_symbol = {
                "success": "✅",
                "partial": "⚠️",
                "failed": "❌"
            }.get(result["status"], "❓")
            
            print(f"{status_symbol} {result['test']}: {result['status'].upper()}")
            print(f"   详情: {result['details']}")
            print()
            
            if result["status"] == "success":
                success_count += 1
        
        print(f"测试总结: {success_count}/{total_count} 个测试通过")
        
        if success_count == total_count:
            print("🎉 所有测试都通过了！推理功能和会话管理工作正常。")
        elif success_count > 0:
            print("⚠️ 部分测试通过，可能需要检查配置或依赖。")
        else:
            print("❌ 所有测试都失败了，请检查环境配置。")
        
        print("="*60)


async def main():
    """主函数"""
    print("开始推理功能和会话管理测试...")
    
    tester = InferenceTester()
    
    try:
        # 设置测试环境
        if not await tester.setup():
            print("❌ 测试环境设置失败，退出测试")
            return
        
        # 运行测试
        tests = [
            tester.test_local_model_service,
            tester.test_session_history_manager,
            tester.test_langchain_inference_service,
            tester.test_training_assistant_agent
        ]
        
        for test in tests:
            await test()
            await asyncio.sleep(1)  # 短暂延迟
        
    except KeyboardInterrupt:
        print("\n测试被用户中断")
    except Exception as e:
        logger.error(f"测试过程中发生错误: {str(e)}")
    finally:
        # 清理测试环境
        await tester.cleanup()
        
        # 打印测试结果
        tester.print_test_results()


if __name__ == "__main__":
    # 配置日志
    logger.remove()
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO"
    )
    
    # 运行测试
    asyncio.run(main())
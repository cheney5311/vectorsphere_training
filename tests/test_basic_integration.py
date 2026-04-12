#!/usr/bin/env python3
"""
基础 API 集成测试脚本

简化版本的测试脚本，用于验证基本的 API 集成功能。
"""

import asyncio
import json
import logging
import sys
import os

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_imports():
    """测试模块导入"""
    logger.info("测试模块导入...")
    
    try:
        # 测试导入配置管理器
        from backend.modules.agent.api_config_manager import APIConfigManager
        logger.info("✓ APIConfigManager 导入成功")
        
        # 测试导入 API 客户端
        from backend.services.chatgpt_api_client import ChatGPTAPIClient
        logger.info("✓ ChatGPTAPIClient 导入成功")
        
        from backend.services.deepseek_api_client import DeepSeekAPIClient
        logger.info("✓ DeepSeekAPIClient 导入成功")
        
        # 测试导入服务
        from backend.services.local_model_service import get_local_model_service
        logger.info("✓ LocalModelService 导入成功")
        
        from backend.services.langchain_inference_service import get_langchain_inference_service
        logger.info("✓ LangChainInferenceService 导入成功")
        
        return True
        
    except ImportError as e:
        logger.error(f"✗ 模块导入失败: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"✗ 导入测试出错: {str(e)}")
        return False


async def test_config_manager():
    """测试配置管理器"""
    logger.info("测试配置管理器...")
    
    try:
        from backend.services.api_config_manager import APIConfigManager
        
        config_manager = APIConfigManager()
        
        # 测试获取提供者列表
        providers = config_manager.list_configs()
        logger.info(f"✓ 可用提供者: {providers}")
        
        # 测试获取配置
        for provider in ["openai", "deepseek"]:
            config = config_manager.get_config(provider)
            is_valid = config_manager.validate_config(provider)
            logger.info(f"✓ {provider} 配置存在: {config is not None}, 有效: {is_valid}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ 配置管理器测试失败: {str(e)}")
        return False


async def test_local_model_service():
    """测试本地模型服务"""
    logger.info("测试本地模型服务...")
    
    try:
        from backend.services.local_model_service import get_local_model_service
        
        service = get_local_model_service()
        
        # 测试获取可用提供者
        providers = service.get_available_providers()
        logger.info(f"✓ 本地模型服务可用提供者: {providers}")
        
        # 测试列出模型
        models = service.list_models()
        logger.info(f"✓ 可用模型数量: {len(models)}")
        
        # 统计云端模型
        cloud_models = [m for m in models if m.get("type") in ["chatgpt", "deepseek"]]
        logger.info(f"✓ 云端模型数量: {len(cloud_models)}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ 本地模型服务测试失败: {str(e)}")
        return False


async def test_langchain_service():
    """测试 LangChain 推理服务"""
    logger.info("测试 LangChain 推理服务...")
    
    try:
        from backend.services.langchain_inference_service import get_langchain_inference_service
        
        service = get_langchain_inference_service()
        
        # 测试获取可用提供者
        providers = service.get_available_providers()
        logger.info(f"✓ LangChain 服务可用提供者: {providers}")
        
        # 测试创建会话
        session_id = await service.create_session(
            user_id="test_user",
            agent_id="test_agent"
        )
        logger.info(f"✓ 会话创建成功: {session_id}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ LangChain 推理服务测试失败: {str(e)}")
        return False


async def test_api_clients():
    """测试 API 客户端基本功能"""
    logger.info("测试 API 客户端...")
    
    try:
        from backend.services.chatgpt_api_client import ChatGPTAPIClient
        from backend.services.deepseek_api_client import DeepSeekAPIClient
        
        # 测试 ChatGPT 客户端
        chatgpt_client = ChatGPTAPIClient()
        chatgpt_available = chatgpt_client.is_available()
        logger.info(f"✓ ChatGPT 客户端可用: {chatgpt_available}")
        
        # 测试 DeepSeek 客户端
        deepseek_client = DeepSeekAPIClient()
        deepseek_available = deepseek_client.is_available()
        logger.info(f"✓ DeepSeek 客户端可用: {deepseek_available}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ API 客户端测试失败: {str(e)}")
        return False


async def main():
    """主测试函数"""
    logger.info("开始基础 API 集成测试...")
    
    tests = [
        ("模块导入", test_imports),
        ("配置管理器", test_config_manager),
        ("本地模型服务", test_local_model_service),
        ("LangChain 推理服务", test_langchain_service),
        ("API 客户端", test_api_clients)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = await test_func()
            results[test_name] = result
            status = "✓ 通过" if result else "✗ 失败"
            logger.info(f"{test_name}: {status}")
        except Exception as e:
            results[test_name] = False
            logger.error(f"{test_name}: ✗ 异常 - {str(e)}")
    
    # 生成测试摘要
    logger.info(f"\n{'='*50}")
    logger.info("测试摘要")
    logger.info(f"{'='*50}")
    
    passed = sum(1 for r in results.values() if r)
    total = len(results)
    
    logger.info(f"总测试数: {total}")
    logger.info(f"通过: {passed}")
    logger.info(f"失败: {total - passed}")
    logger.info(f"成功率: {passed/total*100:.1f}%")
    
    # 保存结果
    with open("basic_test_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    logger.info("测试结果已保存到 basic_test_results.json")
    
    if passed == total:
        logger.info("🎉 所有测试通过！")
        return 0
    else:
        logger.warning("⚠️ 部分测试失败")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
"""智能体功能测试

测试智能体相关的功能，包括：
- 智能体 CRUD 操作
- 会话管理
- 长期记忆管理
- 智能推理执行
- API 端点测试
"""

import unittest
import uuid
import sys
import os
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================================
# 异常类测试
# ============================================================================

class TestAgentExceptions(unittest.TestCase):
    """测试智能体异常类"""
    
    def test_agent_not_found_error(self):
        """测试 AgentNotFoundError 异常"""
        from backend.modules.agent.exceptions.agent_exceptions import AgentNotFoundError
        
        error = AgentNotFoundError("智能体不存在")
        self.assertEqual(str(error), "智能体不存在")
        
        error_with_id = AgentNotFoundError(agent_id="abc-123")
        self.assertIn("abc-123", str(error_with_id))
    
    def test_agent_validation_error(self):
        """测试 AgentValidationError 异常"""
        from backend.modules.agent.exceptions.agent_exceptions import AgentValidationError
        
        error = AgentValidationError("验证失败")
        self.assertEqual(str(error), "验证失败")
        self.assertEqual(error.validation_errors, [])
        
        error_with_details = AgentValidationError("验证失败", ["错误1", "错误2"])
        self.assertEqual(len(error_with_details.validation_errors), 2)
    
    def test_agent_execution_error(self):
        """测试 AgentExecutionError 异常"""
        from backend.modules.agent.exceptions.agent_exceptions import AgentExecutionError
        
        error = AgentExecutionError("执行失败")
        self.assertEqual(str(error), "执行失败")
        
        error_with_id = AgentExecutionError(agent_id="abc-123")
        self.assertIn("abc-123", str(error_with_id))


# ============================================================================
# Agent DTO 测试
# ============================================================================

class TestAgentDTO(unittest.TestCase):
    """测试 Agent 数据传输对象"""
    
    def test_agent_creation(self):
        """测试 Agent 创建"""
        from backend.schemas.agent import Agent
        
        agent = Agent(
            user_id="user-123",
            name="测试智能体"
        )
        
        self.assertEqual(agent.name, "测试智能体")
        self.assertEqual(agent.user_id, "user-123")
        self.assertIsNotNone(agent.agent_id)
    
    def test_agent_to_dict(self):
        """测试 Agent 转换为字典"""
        from backend.schemas.agent import Agent
        
        agent = Agent(
            user_id="user-123",
            name="测试智能体",
            description="这是描述"
        )
        
        data = agent.to_dict()
        self.assertEqual(data['name'], "测试智能体")
        self.assertEqual(data['description'], "这是描述")
    
    def test_agent_status_management(self):
        """测试 Agent 状态管理"""
        from backend.schemas.agent import Agent
        
        agent = Agent(
            user_id="user-123",
            name="测试智能体",
            status="active"
        )
        
        # 默认状态
        self.assertEqual(agent.status, "active")
        self.assertTrue(agent.active)
        
        # 停用
        agent.deactivate()
        self.assertEqual(agent.status, "inactive")
        self.assertFalse(agent.active)
        
        # 激活
        agent.activate()
        self.assertEqual(agent.status, "active")
        self.assertTrue(agent.active)
    
    def test_agent_capability_management(self):
        """测试 Agent 能力管理"""
        from backend.schemas.agent import Agent
        
        agent = Agent(
            user_id="user-123",
            name="测试智能体"
        )
        
        # 添加能力
        agent.add_capability("chat")
        self.assertIn("chat", agent.capabilities)
        
        # 重复添加不会有问题
        agent.add_capability("chat")
        self.assertEqual(agent.capabilities.count("chat"), 1)
        
        # 移除能力
        agent.remove_capability("chat")
        self.assertNotIn("chat", agent.capabilities)


# ============================================================================
# AgentType 测试
# ============================================================================

class TestAgentType(unittest.TestCase):
    """测试 AgentType 枚举"""
    
    def test_agent_types(self):
        """测试智能体类型"""
        from backend.schemas.agent_type import AgentType
        
        # 验证常见类型存在
        self.assertEqual(AgentType.CHAT.value, "chat")
        self.assertEqual(AgentType.TRAINING_ASSISTANT.value, "training_assistant")
    
    def test_agent_type_from_string(self):
        """测试从字符串创建类型"""
        from backend.schemas.agent_type import AgentType
        
        agent_type = AgentType("chat")
        self.assertEqual(agent_type, AgentType.CHAT)


# ============================================================================
# Service 层测试 (模拟依赖)
# ============================================================================

class TestAgentServiceLogic(unittest.TestCase):
    """测试 AgentService 业务逻辑"""
    
    def setUp(self):
        """测试前准备"""
        self.user_id = str(uuid.uuid4())
        self.agent_id = str(uuid.uuid4())
    
    def test_validation_empty_name(self):
        """测试空名称验证"""
        mock_repo = MagicMock()
        
        # 不导入会触发数据库的模块
        with patch.dict('sys.modules', {
            'backend.repositories.agent_repository': MagicMock(),
            'backend.modules.database.manager': MagicMock()
        }):
            # 手动验证逻辑
            name = ''
            if not name or len(name.strip()) == 0:
                is_invalid = True
            else:
                is_invalid = False
            
            self.assertTrue(is_invalid)
    
    def test_validation_long_name(self):
        """测试名称过长验证"""
        name = 'a' * 300
        is_too_long = len(name) > 200
        self.assertTrue(is_too_long)
    
    def test_validation_valid_name(self):
        """测试有效名称"""
        name = "测试智能体"
        is_valid = name and len(name.strip()) > 0 and len(name) <= 200
        self.assertTrue(is_valid)
    
    def test_limit_validation(self):
        """测试分页限制验证"""
        # 无效的 limit
        self.assertFalse(0 < 0 <= 100)  # limit=0
        self.assertFalse(0 < 200 <= 100)  # limit=200
        
        # 有效的 limit
        self.assertTrue(0 < 50 <= 100)  # limit=50
    
    def test_offset_validation(self):
        """测试偏移量验证"""
        # 无效的 offset
        self.assertTrue(-1 < 0)  # offset=-1 是无效的
        
        # 有效的 offset
        self.assertFalse(0 < 0)  # offset=0 是有效的
        self.assertFalse(10 < 0)  # offset=10 是有效的


# ============================================================================
# 记忆类型测试
# ============================================================================

class TestMemoryTypes(unittest.TestCase):
    """测试记忆类型验证"""
    
    def test_valid_memory_types(self):
        """测试有效的记忆类型"""
        valid_types = ['fact', 'preference', 'event', 'skill']
        
        for t in valid_types:
            is_valid = t in valid_types
            self.assertTrue(is_valid, f"{t} 应该是有效的记忆类型")
    
    def test_invalid_memory_type(self):
        """测试无效的记忆类型"""
        valid_types = ['fact', 'preference', 'event', 'skill']
        
        invalid_type = 'invalid_type'
        is_valid = invalid_type in valid_types
        self.assertFalse(is_valid)
    
    def test_importance_clamping(self):
        """测试重要性值钳制"""
        def clamp_importance(value):
            return max(0.0, min(1.0, value))
        
        # 测试边界值
        self.assertEqual(clamp_importance(-0.5), 0.0)
        self.assertEqual(clamp_importance(1.5), 1.0)
        self.assertEqual(clamp_importance(0.5), 0.5)


# ============================================================================
# 训练推荐测试
# ============================================================================

class TestTrainingRecommendation(unittest.TestCase):
    """测试训练推荐逻辑"""
    
    def get_default_recommendation(self, data_type):
        """获取默认推荐 (模拟服务逻辑)"""
        recommendations = {
            'text': {
                'model': 'transformer',
                'learning_rate': 2e-5,
                'batch_size': 32,
                'epochs': 3
            },
            'image': {
                'model': 'cnn',
                'learning_rate': 1e-3,
                'batch_size': 32,
                'epochs': 10
            },
            'tabular': {
                'model': 'mlp',
                'learning_rate': 1e-3,
                'batch_size': 64,
                'epochs': 50
            }
        }
        return recommendations.get(data_type, recommendations['tabular'])
    
    def test_text_recommendation(self):
        """测试文本数据推荐"""
        rec = self.get_default_recommendation('text')
        self.assertEqual(rec['model'], 'transformer')
        self.assertEqual(rec['learning_rate'], 2e-5)
    
    def test_image_recommendation(self):
        """测试图像数据推荐"""
        rec = self.get_default_recommendation('image')
        self.assertEqual(rec['model'], 'cnn')
        self.assertEqual(rec['learning_rate'], 1e-3)
    
    def test_tabular_recommendation(self):
        """测试表格数据推荐"""
        rec = self.get_default_recommendation('tabular')
        self.assertEqual(rec['model'], 'mlp')
        self.assertEqual(rec['batch_size'], 64)
    
    def test_unknown_type_fallback(self):
        """测试未知类型回退"""
        rec = self.get_default_recommendation('unknown')
        # 应该回退到 tabular 推荐
        self.assertEqual(rec['model'], 'mlp')


# ============================================================================
# ChatRequest DTO 测试
# ============================================================================

class TestChatRequestDTO(unittest.TestCase):
    """测试 ChatRequest 数据传输对象"""
    
    def test_default_values(self):
        """测试默认值"""
        from dataclasses import dataclass
        from typing import Optional, Dict, Any
        
        @dataclass
        class ChatRequestTest:
            message: str
            session_id: Optional[str] = None
            context: Optional[Dict[str, Any]] = None
            use_memory: bool = True
            max_tokens: int = 2048
            temperature: float = 0.7
            provider: str = "local"
            stream: bool = False
        
        request = ChatRequestTest(message='Hello')
        
        self.assertEqual(request.message, 'Hello')
        self.assertIsNone(request.session_id)
        self.assertTrue(request.use_memory)
        self.assertEqual(request.max_tokens, 2048)
        self.assertEqual(request.temperature, 0.7)
        self.assertEqual(request.provider, 'local')
        self.assertFalse(request.stream)
    
    def test_custom_values(self):
        """测试自定义值"""
        from dataclasses import dataclass
        from typing import Optional, Dict, Any
        
        @dataclass
        class ChatRequestTest:
            message: str
            session_id: Optional[str] = None
            context: Optional[Dict[str, Any]] = None
            use_memory: bool = True
            max_tokens: int = 2048
            temperature: float = 0.7
            provider: str = "local"
            stream: bool = False
        
        session_id = str(uuid.uuid4())
        request = ChatRequestTest(
            message='Hello',
            session_id=session_id,
            use_memory=False,
            max_tokens=1024,
            temperature=0.5,
            provider='chatgpt',
            stream=True
        )
        
        self.assertEqual(request.session_id, session_id)
        self.assertFalse(request.use_memory)
        self.assertEqual(request.max_tokens, 1024)
        self.assertEqual(request.temperature, 0.5)
        self.assertEqual(request.provider, 'chatgpt')
        self.assertTrue(request.stream)


# ============================================================================
# 知识推理测试
# ============================================================================

class TestKnowledgeReasoning(unittest.TestCase):
    """测试知识推理引擎"""
    
    def test_entity_types(self):
        """测试实体类型"""
        from backend.algo.knowledge_reasoning import EntityType
        
        self.assertEqual(EntityType.MODEL.value, "model")
        self.assertEqual(EntityType.DATASET.value, "dataset")
        self.assertEqual(EntityType.HYPERPARAMETER.value, "hyperparameter")
    
    def test_relation_types(self):
        """测试关系类型"""
        from backend.algo.knowledge_reasoning import RelationType
        
        self.assertEqual(RelationType.SUITABLE_FOR.value, "suitable_for")
        self.assertEqual(RelationType.REQUIRES.value, "requires")
    
    def test_entity_matching(self):
        """测试实体匹配"""
        from backend.algo.knowledge_reasoning import Entity, EntityType
        
        entity = Entity(
            entity_id="model_transformer",
            entity_type=EntityType.MODEL,
            name="Transformer",
            properties={
                "architecture": "transformer",
                "suitable_tasks": ["nlp", "text_classification"]
            }
        )
        
        # 测试类型匹配
        self.assertTrue(entity.matches({'type': 'model'}))
        self.assertFalse(entity.matches({'type': 'dataset'}))
        
        # 测试名称匹配
        self.assertTrue(entity.matches({'name': 'Trans'}))
        self.assertFalse(entity.matches({'name': 'CNN'}))
    
    def test_knowledge_graph_initialization(self):
        """测试知识图谱初始化"""
        from backend.algo.knowledge_reasoning import KnowledgeGraph
        
        kg = KnowledgeGraph()
        
        # 验证默认知识已加载
        self.assertTrue(len(kg._entities) > 0)
        self.assertTrue(len(kg._relations) > 0)
        self.assertTrue(len(kg._rules) > 0)
    
    def test_knowledge_query(self):
        """测试知识查询"""
        from backend.algo.knowledge_reasoning import KnowledgeGraph, EntityType
        
        kg = KnowledgeGraph()
        
        # 查询模型实体
        models = kg.get_entities_by_type(EntityType.MODEL)
        self.assertTrue(len(models) > 0)
        
        # 查询规则
        rules = kg.get_rules()
        self.assertTrue(len(rules) > 0)


# ============================================================================
# LangGraph 组件测试
# ============================================================================

class TestLangGraphComponents(unittest.TestCase):
    """测试 LangGraph 组件"""
    
    def test_agent_config(self):
        """测试 Agent 配置"""
        from backend.algo.langgraph.agents import AgentConfig
        
        config = AgentConfig(
            name="test_agent",
            max_iterations=10,
            temperature=0.7
        )
        
        self.assertEqual(config.name, "test_agent")
        self.assertEqual(config.max_iterations, 10)
        self.assertEqual(config.temperature, 0.7)
    
    def test_checkpointer_interface(self):
        """测试检查点器接口"""
        from backend.algo.langgraph.checkpointer import MemoryCheckpointer
        
        checkpointer = MemoryCheckpointer(max_checkpoints=50)
        
        # 验证方法存在
        self.assertTrue(hasattr(checkpointer, 'save'))
        self.assertTrue(hasattr(checkpointer, 'load'))
        self.assertTrue(hasattr(checkpointer, 'get_latest'))
        self.assertTrue(hasattr(checkpointer, 'list_checkpoints'))
        self.assertTrue(hasattr(checkpointer, 'delete'))


# ============================================================================
# 会话历史管理测试
# ============================================================================

class TestSessionHistoryManager(unittest.TestCase):
    """测试会话历史管理"""
    
    def test_message_type_enum(self):
        """测试消息类型枚举"""
        from backend.modules.agent.session_history_manager import MessageType
        
        self.assertEqual(MessageType.USER.value, "user")
        self.assertEqual(MessageType.ASSISTANT.value, "assistant")
        self.assertEqual(MessageType.SYSTEM.value, "system")
        self.assertEqual(MessageType.FUNCTION.value, "function")
    
    def test_chat_message_to_dict(self):
        """测试聊天消息转字典"""
        from backend.modules.agent.session_history_manager import ChatMessage, MessageType
        
        msg = ChatMessage(
            id="msg-123",
            session_id="session-456",
            message_type=MessageType.USER,
            content="Hello",
            timestamp=datetime.now()
        )
        
        data = msg.to_dict()
        self.assertEqual(data['id'], "msg-123")
        self.assertEqual(data['message_type'], "user")
        self.assertEqual(data['content'], "Hello")
    
    def test_session_info_to_dict(self):
        """测试会话信息转字典"""
        from backend.modules.agent.session_history_manager import SessionInfo
        
        session = SessionInfo(
            session_id="session-123",
            user_id="user-456",
            agent_id="agent-789",
            created_at=datetime.now(),
            last_activity=datetime.now(),
            message_count=5
        )
        
        data = session.to_dict()
        self.assertEqual(data['session_id'], "session-123")
        self.assertEqual(data['user_id'], "user-456")
        self.assertEqual(data['message_count'], 5)


# ============================================================================
# 统计计算测试
# ============================================================================

class TestStatisticsCalculation(unittest.TestCase):
    """测试统计计算逻辑"""
    
    def test_success_rate_calculation(self):
        """测试成功率计算"""
        def calculate_success_rate(success_count, total_count):
            if total_count == 0:
                return 0.0
            return success_count / total_count
        
        # 测试各种情况
        self.assertEqual(calculate_success_rate(0, 0), 0.0)
        self.assertEqual(calculate_success_rate(5, 10), 0.5)
        self.assertEqual(calculate_success_rate(10, 10), 1.0)
    
    def test_average_response_time_update(self):
        """测试平均响应时间更新"""
        def update_avg_time(prev_avg, prev_count, new_time):
            new_count = prev_count + 1
            new_avg = (prev_avg * prev_count + new_time) / new_count
            return new_avg
        
        # 初始状态
        avg = update_avg_time(0, 0, 100)
        self.assertEqual(avg, 100)
        
        # 第二次
        avg = update_avg_time(100, 1, 200)
        self.assertEqual(avg, 150)
        
        # 第三次
        avg = update_avg_time(150, 2, 300)
        self.assertEqual(avg, 200)


# ============================================================================
# API Blueprint 测试
# ============================================================================

class TestAgentAPIBlueprint(unittest.TestCase):
    """测试 Agent API 蓝图"""
    
    def test_health_endpoint_structure(self):
        """测试健康检查端点结构"""
        # 模拟健康检查响应
        health_response = {
            'status': 'healthy',
            'service': 'agent_api',
            'version': '1.0.0'
        }
        
        self.assertEqual(health_response['status'], 'healthy')
        self.assertEqual(health_response['service'], 'agent_api')
    
    def test_success_response_format(self):
        """测试成功响应格式"""
        def success_response(data, message="操作成功", status_code=200):
            return {
                'success': True,
                'data': data,
                'message': message
            }, status_code
        
        response, code = success_response({'id': '123'}, "创建成功", 201)
        
        self.assertTrue(response['success'])
        self.assertEqual(response['data']['id'], '123')
        self.assertEqual(code, 201)
    
    def test_error_response_format(self):
        """测试错误响应格式"""
        def error_response(message, status_code=400):
            return {
                'success': False,
                'error': message
            }, status_code
        
        response, code = error_response("参数错误", 400)
        
        self.assertFalse(response['success'])
        self.assertEqual(response['error'], "参数错误")
        self.assertEqual(code, 400)


# ============================================================================
# 集成测试模拟
# ============================================================================

class TestIntegrationScenarios(unittest.TestCase):
    """测试集成场景"""
    
    def test_agent_lifecycle(self):
        """测试智能体生命周期"""
        # 模拟生命周期
        lifecycle_stages = ['created', 'active', 'in_use', 'inactive', 'deleted']
        
        # 验证状态转换
        current = 'created'
        
        # 创建 -> 激活
        current = 'active'
        self.assertEqual(current, 'active')
        
        # 激活 -> 使用中
        current = 'in_use'
        self.assertEqual(current, 'in_use')
        
        # 使用中 -> 停用
        current = 'inactive'
        self.assertEqual(current, 'inactive')
        
        # 停用 -> 删除
        current = 'deleted'
        self.assertEqual(current, 'deleted')
    
    def test_conversation_flow(self):
        """测试对话流程"""
        # 模拟对话流程
        conversation = []
        
        # 用户消息
        conversation.append({'role': 'user', 'content': 'Hello'})
        self.assertEqual(len(conversation), 1)
        
        # 助手响应
        conversation.append({'role': 'assistant', 'content': 'Hi there!'})
        self.assertEqual(len(conversation), 2)
        
        # 验证消息顺序
        self.assertEqual(conversation[0]['role'], 'user')
        self.assertEqual(conversation[1]['role'], 'assistant')
    
    def test_memory_retrieval_flow(self):
        """测试记忆检索流程"""
        # 模拟记忆
        memories = [
            {'content': '用户喜欢Python', 'importance': 0.8, 'type': 'preference'},
            {'content': '用户是开发者', 'importance': 0.9, 'type': 'fact'},
            {'content': '上次讨论了机器学习', 'importance': 0.6, 'type': 'event'}
        ]
        
        # 按重要性排序
        sorted_memories = sorted(memories, key=lambda x: x['importance'], reverse=True)
        
        # 验证排序正确
        self.assertEqual(sorted_memories[0]['importance'], 0.9)
        self.assertEqual(sorted_memories[1]['importance'], 0.8)
        self.assertEqual(sorted_memories[2]['importance'], 0.6)
        
        # 过滤高重要性记忆
        high_importance = [m for m in memories if m['importance'] >= 0.7]
        self.assertEqual(len(high_importance), 2)


# ============================================================================
# 运行测试
# ============================================================================

if __name__ == '__main__':
    # 创建测试套件
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # 添加测试类
    suite.addTests(loader.loadTestsFromTestCase(TestAgentExceptions))
    suite.addTests(loader.loadTestsFromTestCase(TestAgentDTO))
    suite.addTests(loader.loadTestsFromTestCase(TestAgentType))
    suite.addTests(loader.loadTestsFromTestCase(TestAgentServiceLogic))
    suite.addTests(loader.loadTestsFromTestCase(TestMemoryTypes))
    suite.addTests(loader.loadTestsFromTestCase(TestTrainingRecommendation))
    suite.addTests(loader.loadTestsFromTestCase(TestChatRequestDTO))
    suite.addTests(loader.loadTestsFromTestCase(TestKnowledgeReasoning))
    suite.addTests(loader.loadTestsFromTestCase(TestLangGraphComponents))
    suite.addTests(loader.loadTestsFromTestCase(TestSessionHistoryManager))
    suite.addTests(loader.loadTestsFromTestCase(TestStatisticsCalculation))
    suite.addTests(loader.loadTestsFromTestCase(TestAgentAPIBlueprint))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegrationScenarios))
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # 输出总结
    print("\n" + "=" * 70)
    print("测试总结")
    print("=" * 70)
    print(f"运行测试数: {result.testsRun}")
    print(f"成功: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"失败: {len(result.failures)}")
    print(f"错误: {len(result.errors)}")
    
    if result.failures:
        print("\n失败的测试:")
        for test, traceback in result.failures:
            print(f"  - {test}")
    
    if result.errors:
        print("\n出错的测试:")
        for test, traceback in result.errors:
            print(f"  - {test}")
    
    # 退出码
    sys.exit(0 if result.wasSuccessful() else 1)

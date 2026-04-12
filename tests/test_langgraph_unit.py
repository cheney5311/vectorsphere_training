"""
LangGraph 模块单元测试

测试 backend/algo/langgraph 下各个模块的功能方法
"""

import sys
import os
import unittest
import traceback
from datetime import datetime
from typing import Dict, Any, List

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class TestStateModule(unittest.TestCase):
    """测试 state 模块"""
    
    @classmethod
    def setUpClass(cls):
        """导入 state 模块"""
        from backend.algo.langgraph import state
        cls.state = state
    
    def test_agent_status_enum(self):
        """测试 AgentStatus 枚举"""
        AgentStatus = self.state.AgentStatus
        # 检查枚举成员 - 使用正确的枚举值（FAILED 而不是 ERROR）
        self.assertTrue(hasattr(AgentStatus, 'IDLE'))
        self.assertTrue(hasattr(AgentStatus, 'RUNNING'))
        self.assertTrue(hasattr(AgentStatus, 'COMPLETED'))
        self.assertTrue(hasattr(AgentStatus, 'FAILED'))  # 正确的枚举名
        print("  ✓ AgentStatus 枚举测试通过")
    
    def test_message_type_enum(self):
        """测试 MessageType 枚举"""
        MessageType = self.state.MessageType
        self.assertTrue(hasattr(MessageType, 'HUMAN'))
        self.assertTrue(hasattr(MessageType, 'AI'))
        self.assertTrue(hasattr(MessageType, 'SYSTEM'))
        self.assertTrue(hasattr(MessageType, 'TOOL'))
        print("  ✓ MessageType 枚举测试通过")
    
    def test_agent_message_creation(self):
        """测试 AgentMessage 创建"""
        AgentMessage = self.state.AgentMessage
        MessageType = self.state.MessageType
        
        # 创建消息 - 使用正确的参数名 'type' 而不是 'message_type'
        msg = AgentMessage(
            content="Hello, world!",
            type=MessageType.HUMAN,
            name="user"
        )
        self.assertEqual(msg.content, "Hello, world!")
        self.assertEqual(msg.type, MessageType.HUMAN)
        self.assertEqual(msg.name, "user")
        print("  ✓ AgentMessage 创建测试通过")
    
    def test_tool_call_creation(self):
        """测试 ToolCall 创建"""
        ToolCall = self.state.ToolCall
        
        tool_call = ToolCall(
            id="call_123",
            name="search",
            arguments={"query": "Python tutorial"}
        )
        self.assertEqual(tool_call.id, "call_123")
        self.assertEqual(tool_call.name, "search")
        self.assertEqual(tool_call.arguments, {"query": "Python tutorial"})
        print("  ✓ ToolCall 创建测试通过")
    
    def test_tool_result_creation(self):
        """测试 ToolResult 创建"""
        ToolResult = self.state.ToolResult
        
        # 使用正确的参数名 'tool_call_id' 而不是 'call_id'
        result = ToolResult(
            tool_call_id="call_123",
            name="search",
            result="Found 10 results"
        )
        self.assertEqual(result.tool_call_id, "call_123")
        self.assertEqual(result.name, "search")
        self.assertEqual(result.result, "Found 10 results")
        print("  ✓ ToolResult 创建测试通过")
    
    def test_agent_state_creation(self):
        """测试 AgentState 创建"""
        AgentState = self.state.AgentState
        
        state = AgentState()
        self.assertIsNotNone(state)
        self.assertTrue(hasattr(state, 'messages'))
        self.assertTrue(hasattr(state, 'status'))
        print("  ✓ AgentState 创建测试通过")
    
    def test_state_checkpoint_creation(self):
        """测试 StateCheckpoint 创建"""
        StateCheckpoint = self.state.StateCheckpoint
        AgentState = self.state.AgentState
        
        # StateCheckpoint 需要 thread_id 参数，且 state 应为 Dict
        checkpoint = StateCheckpoint(
            checkpoint_id="cp_001",
            thread_id="thread_001",
            state={"messages": [], "status": "idle"}
        )
        self.assertEqual(checkpoint.checkpoint_id, "cp_001")
        self.assertEqual(checkpoint.thread_id, "thread_001")
        self.assertIsNotNone(checkpoint.timestamp)
        print("  ✓ StateCheckpoint 创建测试通过")
    
    def test_memory_entry_creation(self):
        """测试 MemoryEntry 创建"""
        if not hasattr(self.state, 'MemoryEntry'):
            print("  ⚠ MemoryEntry 不可用，跳过")
            return
        
        MemoryEntry = self.state.MemoryEntry
        MemoryType = self.state.MemoryType
        
        entry = MemoryEntry(
            memory_type=MemoryType.SHORT_TERM,
            content="Test memory",
            importance=0.8
        )
        self.assertEqual(entry.content, "Test memory")
        print("  ✓ MemoryEntry 创建测试通过")
    
    def test_plan_step_creation(self):
        """测试 PlanStep 创建"""
        if not hasattr(self.state, 'PlanStep'):
            print("  ⚠ PlanStep 不可用，跳过")
            return
        
        PlanStep = self.state.PlanStep
        
        step = PlanStep(
            step_id="step_001",
            description="First step",
            action="search"
        )
        self.assertEqual(step.step_id, "step_001")
        print("  ✓ PlanStep 创建测试通过")


class TestToolsModule(unittest.TestCase):
    """测试 tools 模块"""
    
    @classmethod
    def setUpClass(cls):
        from backend.algo.langgraph import tools
        cls.tools = tools
    
    def test_tool_parameter_creation(self):
        """测试 ToolParameter 创建"""
        ToolParameter = self.tools.ToolParameter
        
        param = ToolParameter(
            name="query",
            type="string",
            description="Search query",
            required=True
        )
        self.assertEqual(param.name, "query")
        self.assertEqual(param.type, "string")
        self.assertTrue(param.required)
        print("  ✓ ToolParameter 创建测试通过")
    
    def test_tool_creation(self):
        """测试 Tool 创建"""
        Tool = self.tools.Tool
        
        def search_func(query: str) -> str:
            return f"Results for: {query}"
        
        tool = Tool(
            name="search",
            description="Search for information",
            func=search_func
        )
        self.assertEqual(tool.name, "search")
        self.assertIsNotNone(tool.func)
        print("  ✓ Tool 创建测试通过")
    
    def test_tool_decorator(self):
        """测试 @tool 装饰器"""
        tool_decorator = self.tools.tool
        
        @tool_decorator(name="calculator", description="Calculate expressions")
        def calculator(expression: str) -> str:
            return str(eval(expression))
        
        self.assertTrue(hasattr(calculator, '_tool') or callable(calculator))
        print("  ✓ @tool 装饰器测试通过")
    
    def test_tool_registry(self):
        """测试 ToolRegistry"""
        ToolRegistry = self.tools.ToolRegistry
        Tool = self.tools.Tool
        
        registry = ToolRegistry()
        
        def dummy_func():
            pass
        
        tool = Tool(name="dummy", description="Dummy tool", func=dummy_func)
        registry.register(tool)
        
        retrieved = registry.get("dummy")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.name, "dummy")
        print("  ✓ ToolRegistry 测试通过")
    
    def test_tool_executor(self):
        """测试 ToolExecutor"""
        ToolExecutor = self.tools.ToolExecutor
        Tool = self.tools.Tool
        ToolRegistry = self.tools.ToolRegistry
        
        def add_func(a: int, b: int) -> int:
            return a + b
        
        tool = Tool(name="add", description="Add numbers", func=add_func)
        registry = ToolRegistry()
        registry.register(tool)
        
        executor = ToolExecutor(registry=registry)
        self.assertIsNotNone(executor)
        print("  ✓ ToolExecutor 创建测试通过")


class TestNodesModule(unittest.TestCase):
    """测试 nodes 模块"""
    
    @classmethod
    def setUpClass(cls):
        from backend.algo.langgraph import nodes
        cls.nodes = nodes
    
    def test_node_type_enum(self):
        """测试 NodeType 枚举"""
        NodeType = self.nodes.NodeType
        
        self.assertTrue(hasattr(NodeType, 'LLM'))
        self.assertTrue(hasattr(NodeType, 'TOOL'))
        self.assertTrue(hasattr(NodeType, 'ROUTER'))
        self.assertTrue(hasattr(NodeType, 'START'))
        self.assertTrue(hasattr(NodeType, 'END'))
        print("  ✓ NodeType 枚举测试通过")
    
    def test_node_config_creation(self):
        """测试 NodeConfig 创建"""
        NodeConfig = self.nodes.NodeConfig
        NodeType = self.nodes.NodeType
        
        config = NodeConfig(
            name="llm_node",
            node_type=NodeType.LLM,
            description="LLM processing node"
        )
        self.assertEqual(config.name, "llm_node")
        self.assertEqual(config.node_type, NodeType.LLM)
        print("  ✓ NodeConfig 创建测试通过")
    
    def test_base_node_is_abstract(self):
        """测试 BaseNode 是抽象类"""
        BaseNode = self.nodes.BaseNode
        # BaseNode 是抽象类，不能直接实例化
        with self.assertRaises(TypeError):
            BaseNode(config=None)
        print("  ✓ BaseNode 是抽象类测试通过")
    
    def test_start_node_creation(self):
        """测试 StartNode 创建"""
        StartNode = self.nodes.StartNode
        node = StartNode()
        self.assertIsNotNone(node)
        self.assertEqual(node.name, "__start__")
        print("  ✓ StartNode 创建测试通过")
    
    def test_end_node_creation(self):
        """测试 EndNode 创建"""
        EndNode = self.nodes.EndNode
        node = EndNode()
        self.assertIsNotNone(node)
        self.assertEqual(node.name, "__end__")
        print("  ✓ EndNode 创建测试通过")
    
    def test_node_factory(self):
        """测试 NodeFactory"""
        NodeFactory = self.nodes.NodeFactory
        NodeType = self.nodes.NodeType
        
        factory = NodeFactory()
        self.assertIsNotNone(factory)
        # NodeFactory 可能有不同的创建方法
        if hasattr(factory, 'create_llm_node'):
            node = factory.create_llm_node(name="test_llm")
            self.assertIsNotNone(node)
        print("  ✓ NodeFactory 测试通过")


class TestEdgesModule(unittest.TestCase):
    """测试 edges 模块"""
    
    @classmethod
    def setUpClass(cls):
        from backend.algo.langgraph import edges
        cls.edges = edges
    
    def test_edge_type_enum(self):
        """测试 EdgeType 枚举"""
        EdgeType = self.edges.EdgeType
        
        self.assertTrue(hasattr(EdgeType, 'NORMAL'))
        self.assertTrue(hasattr(EdgeType, 'CONDITIONAL'))
        print("  ✓ EdgeType 枚举测试通过")
    
    def test_edge_creation(self):
        """测试 Edge 创建"""
        Edge = self.edges.Edge
        
        edge = Edge(
            source="node_a",
            target="node_b"
        )
        self.assertEqual(edge.source, "node_a")
        self.assertEqual(edge.target, "node_b")
        print("  ✓ Edge 创建测试通过")
    
    def test_conditional_edge_creation(self):
        """测试 ConditionalEdge 创建"""
        ConditionalEdge = self.edges.ConditionalEdge
        
        def condition_func(state):
            return "next_node"
        
        # ConditionalEdge 需要 source 和 target，condition_func 是条件函数
        edge = ConditionalEdge(
            source="router",
            target="default_target",
            condition_func=condition_func
        )
        self.assertEqual(edge.source, "router")
        self.assertEqual(edge.target, "default_target")
        self.assertIsNotNone(edge.condition_func)
        print("  ✓ ConditionalEdge 创建测试通过")
    
    def test_loop_edge_creation(self):
        """测试 LoopEdge 创建"""
        LoopEdge = self.edges.LoopEdge
        
        edge = LoopEdge(
            source="process",
            target="check",
            max_iterations=5
        )
        self.assertEqual(edge.max_iterations, 5)
        print("  ✓ LoopEdge 创建测试通过")
    
    def test_edge_condition_creation(self):
        """测试 EdgeCondition 创建"""
        EdgeCondition = self.edges.EdgeCondition
        
        def my_condition(state):
            return state.get("ready", False)
        
        condition = EdgeCondition(
            name="ready_check",
            condition_func=my_condition
        )
        self.assertEqual(condition.name, "ready_check")
        print("  ✓ EdgeCondition 创建测试通过")
    
    def test_edge_builder(self):
        """测试 EdgeBuilder"""
        EdgeBuilder = self.edges.EdgeBuilder
        
        # EdgeBuilder 需要 source 参数初始化，使用 to() 方法设置目标
        builder = EdgeBuilder(source="start")
        builder.to("process")
        edge = builder.build()
        self.assertEqual(edge.source, "start")
        self.assertEqual(edge.target, "process")
        print("  ✓ EdgeBuilder 测试通过")


class TestGraphModule(unittest.TestCase):
    """测试 graph 模块"""
    
    @classmethod
    def setUpClass(cls):
        from backend.algo.langgraph import graph
        cls.graph = graph
    
    def test_graph_status_enum(self):
        """测试 GraphStatus 枚举"""
        GraphStatus = self.graph.GraphStatus
        
        self.assertTrue(hasattr(GraphStatus, 'BUILDING'))
        self.assertTrue(hasattr(GraphStatus, 'COMPILED'))
        self.assertTrue(hasattr(GraphStatus, 'RUNNING'))
        print("  ✓ GraphStatus 枚举测试通过")
    
    def test_graph_config_creation(self):
        """测试 GraphConfig 创建"""
        GraphConfig = self.graph.GraphConfig
        
        # GraphConfig 使用 max_iterations 而不是 recursion_limit
        config = GraphConfig(
            name="my_graph",
            max_iterations=25
        )
        self.assertEqual(config.name, "my_graph")
        self.assertEqual(config.max_iterations, 25)
        print("  ✓ GraphConfig 创建测试通过")
    
    def test_state_graph_creation(self):
        """测试 StateGraph 创建"""
        StateGraph = self.graph.StateGraph
        GraphConfig = self.graph.GraphConfig
        
        config = GraphConfig(name="test_graph")
        graph = StateGraph(config=config)
        self.assertIsNotNone(graph)
        print("  ✓ StateGraph 创建测试通过")
    
    def test_graph_builder(self):
        """测试 GraphBuilder"""
        GraphBuilder = self.graph.GraphBuilder
        
        builder = GraphBuilder(name="builder_graph")
        self.assertIsNotNone(builder)
        print("  ✓ GraphBuilder 创建测试通过")


class TestCheckpointerModule(unittest.TestCase):
    """测试 checkpointer 模块"""
    
    @classmethod
    def setUpClass(cls):
        from backend.algo.langgraph import checkpointer
        cls.checkpointer = checkpointer
    
    def test_memory_checkpointer_creation(self):
        """测试 MemoryCheckpointer 创建"""
        MemoryCheckpointer = self.checkpointer.MemoryCheckpointer
        
        cp = MemoryCheckpointer()
        self.assertIsNotNone(cp)
        print("  ✓ MemoryCheckpointer 创建测试通过")
    
    def test_create_memory_checkpointer_func(self):
        """测试 create_memory_checkpointer 函数"""
        create_func = self.checkpointer.create_memory_checkpointer
        
        cp = create_func()
        self.assertIsNotNone(cp)
        print("  ✓ create_memory_checkpointer 函数测试通过")
    
    def test_checkpointer_save_and_load(self):
        """测试检查点保存和加载"""
        MemoryCheckpointer = self.checkpointer.MemoryCheckpointer
        
        cp = MemoryCheckpointer()
        
        # 检查点保存需要 AgentState，不是普通字典
        # save 方法签名: save(self, state: AgentState, tags=None, branch="main", ttl=None)
        from backend.algo.langgraph.state import AgentState
        
        state = AgentState(thread_id="test_thread")
        
        # 保存检查点
        checkpoint_id = cp.save(state=state, tags=["test"])
        self.assertIsNotNone(checkpoint_id)
        
        # 加载检查点
        loaded = cp.load(checkpoint_id=checkpoint_id)
        self.assertIsNotNone(loaded)
        print("  ✓ Checkpointer save/load 测试通过")


class TestAgentsModule(unittest.TestCase):
    """测试 agents 模块"""
    
    @classmethod
    def setUpClass(cls):
        from backend.algo.langgraph import agents
        cls.agents = agents
    
    def test_agent_role_creation(self):
        """测试 AgentRole 创建"""
        AgentRole = self.agents.AgentRole
        
        # AgentRole 是 dataclass，不是枚举
        role = AgentRole(
            name="assistant",
            description="A helpful assistant",
            system_prompt="You are a helpful assistant."
        )
        self.assertEqual(role.name, "assistant")
        self.assertEqual(role.description, "A helpful assistant")
        print("  ✓ AgentRole 创建测试通过")
    
    def test_agent_config_creation(self):
        """测试 AgentConfig 创建"""
        AgentConfig = self.agents.AgentConfig
        
        config = AgentConfig(
            name="test_agent",
            max_iterations=10,
            timeout=300.0
        )
        self.assertEqual(config.name, "test_agent")
        self.assertEqual(config.max_iterations, 10)
        print("  ✓ AgentConfig 创建测试通过")
    
    def test_base_agent_is_abstract(self):
        """测试 BaseAgent 是抽象类"""
        BaseAgent = self.agents.BaseAgent
        AgentConfig = self.agents.AgentConfig
        
        # BaseAgent 是抽象类，不能直接实例化
        config = AgentConfig(name="base_test")
        with self.assertRaises(TypeError):
            BaseAgent(config=config)
        print("  ✓ BaseAgent 是抽象类测试通过")
    
    def test_agent_pool_creation(self):
        """测试 AgentPool 创建"""
        AgentPool = self.agents.AgentPool
        
        # AgentPool.__init__ 只接受 max_agents 参数
        pool = AgentPool(max_agents=5)
        self.assertEqual(pool.max_agents, 5)
        print("  ✓ AgentPool 创建测试通过")
    
    def test_agent_orchestrator_creation(self):
        """测试 AgentOrchestrator 创建"""
        AgentOrchestrator = self.agents.AgentOrchestrator
        AgentPool = self.agents.AgentPool
        
        pool = AgentPool(max_agents=10)
        orchestrator = AgentOrchestrator(pool=pool)
        self.assertIsNotNone(orchestrator)
        print("  ✓ AgentOrchestrator 创建测试通过")
    
    def test_agent_registry(self):
        """测试 AgentRegistry"""
        AgentRegistry = self.agents.AgentRegistry
        get_agent_registry = self.agents.get_agent_registry
        
        registry = get_agent_registry()
        self.assertIsNotNone(registry)
        print("  ✓ AgentRegistry 测试通过")


class TestFactoryModule(unittest.TestCase):
    """测试 factory 模块"""
    
    @classmethod
    def setUpClass(cls):
        from backend.algo.langgraph import factory
        cls.factory = factory
    
    def test_agent_factory_creation(self):
        """测试 AgentFactory 创建"""
        AgentFactory = self.factory.AgentFactory
        
        factory = AgentFactory()
        self.assertIsNotNone(factory)
        print("  ✓ AgentFactory 创建测试通过")
    
    def test_master_factory_creation(self):
        """测试 MasterFactory 创建"""
        MasterFactory = self.factory.MasterFactory
        
        factory = MasterFactory()
        self.assertIsNotNone(factory)
        print("  ✓ MasterFactory 创建测试通过")
    
    def test_get_master_factory(self):
        """测试 get_master_factory 函数"""
        get_master_factory = self.factory.get_master_factory
        
        factory = get_master_factory()
        self.assertIsNotNone(factory)
        print("  ✓ get_master_factory 函数测试通过")
    
    def test_master_factory_create_agent_message(self):
        """测试 MasterFactory.create_agent_message"""
        MasterFactory = self.factory.MasterFactory
        
        factory = MasterFactory()
        # create_agent_message 使用 name 而不是 role
        msg = factory.create_agent_message(
            content="Test message",
            name="user"
        )
        self.assertIsNotNone(msg)
        self.assertEqual(msg.content, "Test message")
        print("  ✓ MasterFactory.create_agent_message 测试通过")
    
    def test_master_factory_create_tool_call(self):
        """测试 MasterFactory.create_tool_call"""
        MasterFactory = self.factory.MasterFactory
        
        factory = MasterFactory()
        # create_tool_call 使用 tool_call_id 而不是 id
        tool_call = factory.create_tool_call(
            name="search",
            arguments={"query": "test"},
            tool_call_id="tc_001"
        )
        self.assertIsNotNone(tool_call)
        self.assertEqual(tool_call.name, "search")
        print("  ✓ MasterFactory.create_tool_call 测试通过")
    
    def test_master_factory_create_tool_result(self):
        """测试 MasterFactory.create_tool_result"""
        MasterFactory = self.factory.MasterFactory
        
        factory = MasterFactory()
        # create_tool_result 使用 tool_call_id 而不是 call_id
        result = factory.create_tool_result(
            tool_call_id="tc_001",
            name="search",
            result="Found results"
        )
        self.assertIsNotNone(result)
        print("  ✓ MasterFactory.create_tool_result 测试通过")
    
    def test_master_factory_create_node_config(self):
        """测试 MasterFactory.create_node_config"""
        MasterFactory = self.factory.MasterFactory
        
        factory = MasterFactory()
        config = factory.create_node_config(
            name="test_node",
            node_type="llm"
        )
        self.assertIsNotNone(config)
        print("  ✓ MasterFactory.create_node_config 测试通过")
    
    def test_master_factory_create_edge_condition(self):
        """测试 MasterFactory.create_edge_condition"""
        MasterFactory = self.factory.MasterFactory
        
        factory = MasterFactory()
        
        def my_condition(state):
            return True
        
        condition = factory.create_edge_condition(
            name="test_condition",
            condition_func=my_condition
        )
        self.assertIsNotNone(condition)
        print("  ✓ MasterFactory.create_edge_condition 测试通过")
    
    def test_master_factory_create_graph_metrics(self):
        """测试 MasterFactory.create_graph_metrics"""
        MasterFactory = self.factory.MasterFactory
        
        factory = MasterFactory()
        metrics = factory.create_graph_metrics()
        self.assertIsNotNone(metrics)
        print("  ✓ MasterFactory.create_graph_metrics 测试通过")
    
    def test_master_factory_create_tool_parameter(self):
        """测试 MasterFactory.create_tool_parameter"""
        MasterFactory = self.factory.MasterFactory
        
        factory = MasterFactory()
        # 使用 param_type 而不是 type
        param = factory.create_tool_parameter(
            name="query",
            param_type="string",
            description="Search query",
            required=True
        )
        self.assertIsNotNone(param)
        print("  ✓ MasterFactory.create_tool_parameter 测试通过")
    
    def test_master_factory_create_tool_cache(self):
        """测试 MasterFactory.create_tool_cache"""
        MasterFactory = self.factory.MasterFactory
        
        factory = MasterFactory()
        cache = factory.create_tool_cache(
            max_size=100,
            default_ttl=300
        )
        self.assertIsNotNone(cache)
        print("  ✓ MasterFactory.create_tool_cache 测试通过")
    
    def test_master_factory_create_retry_handler(self):
        """测试 MasterFactory.create_retry_handler"""
        MasterFactory = self.factory.MasterFactory
        
        factory = MasterFactory()
        handler = factory.create_retry_handler(
            max_retries=3,
            base_delay=1.0
        )
        self.assertIsNotNone(handler)
        print("  ✓ MasterFactory.create_retry_handler 测试通过")
    
    def test_master_factory_health_check(self):
        """测试 MasterFactory.health_check"""
        MasterFactory = self.factory.MasterFactory
        
        factory = MasterFactory()
        health = factory.health_check()
        self.assertIsNotNone(health)
        self.assertIn("status", health)
        print("  ✓ MasterFactory.health_check 测试通过")
    
    def test_master_factory_create_execution_event(self):
        """测试 MasterFactory.create_execution_event"""
        MasterFactory = self.factory.MasterFactory
        
        factory = MasterFactory()
        # create_execution_event 需要 event_type 和 node_name
        event = factory.create_execution_event(
            event_type="start",
            node_name="test_node"
        )
        self.assertIsNotNone(event)
        print("  ✓ MasterFactory.create_execution_event 测试通过")
    
    def test_master_factory_create_tool_rate_limiter(self):
        """测试 MasterFactory.create_tool_rate_limiter"""
        MasterFactory = self.factory.MasterFactory
        
        factory = MasterFactory()
        # 使用正确的参数 rate, burst, per_tool
        limiter = factory.create_tool_rate_limiter(
            rate=10.0,
            burst=20,
            per_tool=True
        )
        self.assertIsNotNone(limiter)
        print("  ✓ MasterFactory.create_tool_rate_limiter 测试通过")
    
    def test_master_factory_create_checkpoint_tag(self):
        """测试 MasterFactory.create_checkpoint_tag"""
        MasterFactory = self.factory.MasterFactory
        
        factory = MasterFactory()
        # create_checkpoint_tag 需要 name 和 checkpoint_id
        tag = factory.create_checkpoint_tag(
            name="important",
            checkpoint_id="cp_001"
        )
        self.assertIsNotNone(tag)
        print("  ✓ MasterFactory.create_checkpoint_tag 测试通过")
    
    def test_master_factory_create_enhanced_checkpoint(self):
        """测试 MasterFactory.create_enhanced_checkpoint"""
        MasterFactory = self.factory.MasterFactory
        
        factory = MasterFactory()
        checkpoint = factory.create_enhanced_checkpoint(
            checkpoint_id="cp_001",
            thread_id="thread_001",
            state={"messages": [], "status": "idle"}
        )
        self.assertIsNotNone(checkpoint)
        print("  ✓ MasterFactory.create_enhanced_checkpoint 测试通过")
    
    def test_master_factory_get_checkpointer_instance(self):
        """测试 MasterFactory.get_checkpointer_instance"""
        MasterFactory = self.factory.MasterFactory
        
        factory = MasterFactory()
        checkpointer = factory.get_checkpointer_instance()
        self.assertIsNotNone(checkpointer)
        print("  ✓ MasterFactory.get_checkpointer_instance 测试通过")


class TestBuiltinToolsModule(unittest.TestCase):
    """测试 builtin_tools 模块"""
    
    @classmethod
    def setUpClass(cls):
        from backend.algo.langgraph import builtin_tools
        cls.builtin_tools = builtin_tools
    
    def test_get_builtin_tools(self):
        """测试 get_builtin_tools 函数"""
        get_builtin_tools = self.builtin_tools.get_builtin_tools
        
        tools = get_builtin_tools()
        self.assertIsInstance(tools, list)
        print(f"  ✓ get_builtin_tools 返回 {len(tools)} 个工具")
    
    def test_get_search_tools(self):
        """测试 get_search_tools 函数"""
        get_search_tools = self.builtin_tools.get_search_tools
        
        tools = get_search_tools()
        self.assertIsInstance(tools, list)
        print(f"  ✓ get_search_tools 返回 {len(tools)} 个工具")
    
    def test_get_code_tools(self):
        """测试 get_code_tools 函数"""
        get_code_tools = self.builtin_tools.get_code_tools
        
        tools = get_code_tools()
        self.assertIsInstance(tools, list)
        print(f"  ✓ get_code_tools 返回 {len(tools)} 个工具")
    
    def test_get_data_tools(self):
        """测试 get_data_tools 函数"""
        get_data_tools = self.builtin_tools.get_data_tools
        
        tools = get_data_tools()
        self.assertIsInstance(tools, list)
        print(f"  ✓ get_data_tools 返回 {len(tools)} 个工具")
    
    def test_get_system_tools(self):
        """测试 get_system_tools 函数"""
        get_system_tools = self.builtin_tools.get_system_tools
        
        tools = get_system_tools()
        self.assertIsInstance(tools, list)
        print(f"  ✓ get_system_tools 返回 {len(tools)} 个工具")


def run_tests():
    """运行所有测试"""
    print("=" * 60)
    print("LangGraph 模块单元测试")
    print("=" * 60)
    
    test_classes = [
        ("State 模块", TestStateModule),
        ("Tools 模块", TestToolsModule),
        ("Nodes 模块", TestNodesModule),
        ("Edges 模块", TestEdgesModule),
        ("Graph 模块", TestGraphModule),
        ("Checkpointer 模块", TestCheckpointerModule),
        ("Agents 模块", TestAgentsModule),
        ("Factory 模块", TestFactoryModule),
        ("Builtin Tools 模块", TestBuiltinToolsModule),
    ]
    
    results = {}
    errors = []
    
    for name, test_class in test_classes:
        print(f"\n{'=' * 40}")
        print(f"测试 {name}")
        print("=" * 40)
        
        try:
            # 创建测试套件
            suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
            
            # 运行测试
            runner = unittest.TextTestRunner(verbosity=0)
            result = runner.run(suite)
            
            passed = result.testsRun - len(result.failures) - len(result.errors)
            results[name] = {
                "total": result.testsRun,
                "passed": passed,
                "failed": len(result.failures),
                "errors": len(result.errors)
            }
            
            # 收集错误
            for test, tb in result.failures + result.errors:
                errors.append({
                    "module": name,
                    "test": str(test),
                    "traceback": tb
                })
                
        except Exception as e:
            print(f"  ✗ 模块测试失败: {e}")
            traceback.print_exc()
            results[name] = {"error": str(e)}
            errors.append({
                "module": name,
                "test": "setup",
                "traceback": traceback.format_exc()
            })
    
    # 打印汇总
    print("\n" + "=" * 60)
    print("测试汇总")
    print("=" * 60)
    
    total_tests = 0
    total_passed = 0
    total_failed = 0
    total_errors = 0
    
    for name, res in results.items():
        if "error" in res:
            print(f"  {name}: 模块级错误 - {res['error']}")
        else:
            total_tests += res["total"]
            total_passed += res["passed"]
            total_failed += res["failed"]
            total_errors += res["errors"]
            status = "✓" if res["failed"] == 0 and res["errors"] == 0 else "✗"
            print(f"  {status} {name}: {res['passed']}/{res['total']} 通过")
    
    print(f"\n总计: {total_passed}/{total_tests} 测试通过")
    print(f"失败: {total_failed}, 错误: {total_errors}")
    
    if errors:
        print("\n" + "=" * 60)
        print("错误详情")
        print("=" * 60)
        for err in errors:
            print(f"\n模块: {err['module']}")
            print(f"测试: {err['test']}")
            print("Traceback:")
            print(err['traceback'])
    
    return len(errors) == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)

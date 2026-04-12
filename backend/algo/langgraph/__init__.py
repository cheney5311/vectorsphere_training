"""
LangGraph 集成模块

提供基于 LangGraph 的有状态 LLM 应用构建能力：
- 多步骤推理 (Multi-step Reasoning)
- 工具调用 (Tool Calling)
- 循环控制 (Loop Control)
- ReAct Agent
- 人机协作 (Human-in-the-loop)

核心组件:
- AgentState: Agent 状态管理
- Tool: 工具定义和注册
- GraphBuilder: 图构建器
- AgentExecutor: Agent 执行器

使用示例:
    ```python
    from backend.algo.langgraph import (
        create_react_agent,
        tool,
        AgentState
    )
    
    # 定义工具
    @tool(name="search", description="搜索信息")
    def search(query: str) -> str:
        return f"搜索结果: {query}"
    
    # 创建 Agent
    agent = create_react_agent(
        name="my_agent",
        tools=[search._tool]
    )
    
    # 执行
    result = agent.invoke("帮我搜索 Python 教程")
    print(result.final_answer)
    ```
"""

from .state import (
    AgentState,
    AgentStatus,
    MessageType,
    AgentMessage,
    ToolCall,
    ToolResult,
    StateCheckpoint,
    StateReducer,
    add_messages
)

from .tools import (
    Tool,
    ToolParameter,
    ToolCategory,
    ToolRegistry,
    ToolExecutor,
    tool,
    async_tool,
    get_global_registry
)

from .nodes import (
    BaseNode,
    NodeType,
    NodeConfig,
    LLMNode,
    ToolNode,
    ConditionalNode,
    HumanNode,
    RouterNode,
    AggregatorNode,
    StartNode,
    EndNode,
    NodeFactory
)

from .edges import (
    Edge,
    EdgeType,
    ConditionalEdge,
    LoopEdge,
    ParallelEdge,
    EdgeCondition,
    EdgeConditions,
    EdgeBuilder,
    edge_from,
    should_continue_condition,
    route_by_tool_calls,
    route_after_agent,
    route_after_tools
)

from .graph import (
    StateGraph,
    GraphConfig,
    GraphStatus,
    GraphBuilder,
    CompiledGraph,
    Subgraph,
    create_simple_graph
)

from .agents import (
    BaseAgent,
    AgentConfig,
    AgentRole,
    ReActAgent,
    PlanAndExecuteAgent,
    ReflexionAgent,
    MultiAgentSystem,
    # 工业级编排组件
    AgentPool,
    AgentOrchestrator,
    AgentRegistry,
    get_agent_registry
)

from .checkpointer import (
    Checkpointer,
    MemoryCheckpointer,
    RedisCheckpointer,
    create_memory_checkpointer,
    create_redis_checkpointer
)

from .factory import (
    AgentFactory,
    MasterFactory,
    get_master_factory,
    create_react_agent,
    create_plan_execute_agent,
    create_multi_agent_system,
    create_research_agent,
    create_coding_agent,
    create_data_analyst_agent,
    create_ml_training_agent
)

from .builtin_tools import (
    get_builtin_tools,
    get_tools_by_category,
    get_search_tools,
    get_code_tools,
    get_data_tools,
    get_system_tools
)

__all__ = [
    # 状态
    'AgentState',
    'AgentStatus',
    'MessageType',
    'AgentMessage',
    'ToolCall',
    'ToolResult',
    'StateCheckpoint',
    'StateReducer',
    'add_messages',
    
    # 工具
    'Tool',
    'ToolParameter',
    'ToolCategory',
    'ToolRegistry',
    'ToolExecutor',
    'tool',
    'async_tool',
    'get_global_registry',
    
    # 节点
    'BaseNode',
    'NodeType',
    'NodeConfig',
    'LLMNode',
    'ToolNode',
    'ConditionalNode',
    'HumanNode',
    'RouterNode',
    'AggregatorNode',
    'StartNode',
    'EndNode',
    'NodeFactory',
    
    # 边
    'Edge',
    'EdgeType',
    'ConditionalEdge',
    'LoopEdge',
    'ParallelEdge',
    'EdgeCondition',
    'EdgeConditions',
    'EdgeBuilder',
    'edge_from',
    'should_continue_condition',
    'route_by_tool_calls',
    'route_after_agent',
    'route_after_tools',
    
    # 图
    'StateGraph',
    'GraphConfig',
    'GraphStatus',
    'GraphBuilder',
    'CompiledGraph',
    'Subgraph',
    'create_simple_graph',
    
    # Agent
    'BaseAgent',
    'AgentConfig',
    'AgentRole',
    'ReActAgent',
    'PlanAndExecuteAgent',
    'ReflexionAgent',
    'MultiAgentSystem',
    
    # 工业级编排组件
    'AgentPool',
    'AgentOrchestrator',
    'AgentRegistry',
    'get_agent_registry',
    
    # 检查点
    'Checkpointer',
    'MemoryCheckpointer',
    'RedisCheckpointer',
    'create_memory_checkpointer',
    'create_redis_checkpointer',
    
    # 工厂
    'AgentFactory',
    'MasterFactory',
    'get_master_factory',
    'create_react_agent',
    'create_plan_execute_agent',
    'create_multi_agent_system',
    'create_research_agent',
    'create_coding_agent',
    'create_data_analyst_agent',
    'create_ml_training_agent',
    
    # 内置工具
    'get_builtin_tools',
    'get_tools_by_category',
    'get_search_tools',
    'get_code_tools',
    'get_data_tools',
    'get_system_tools'
]



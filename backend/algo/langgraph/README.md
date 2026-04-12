# LangGraph Agent 模块

基于状态图的有状态 LLM 应用构建框架，提供多步骤推理、工具调用和循环控制能力。

## 特性

- ✅ **多步骤推理**: ReAct 循环实现思考-行动-观察
- ✅ **工具调用**: 装饰器定义，自动参数推断
- ✅ **循环控制**: 条件边、最大迭代、终止条件
- ✅ **状态持久化**: Memory/Redis 检查点
- ✅ **流式执行**: 逐步返回中间状态
- ✅ **人机协作**: HumanNode 支持人工介入
- ✅ **多 Agent**: Supervisor/轮询协作模式

---

## 快速开始

### 1. 定义工具

```python
from backend.algo.langgraph import tool, ToolCategory

@tool(
    name="search",
    description="搜索信息",
    category=ToolCategory.SEARCH
)
def search(query: str) -> str:
    """搜索工具"""
    return f"搜索结果: {query}"
```

### 2. 创建 Agent

```python
from backend.algo.langgraph import create_react_agent

agent = create_react_agent(
    name="assistant",
    tools=[search._tool],
    max_iterations=10
)
```

### 3. 执行任务

```python
# 同步执行
result = agent.invoke("帮我搜索 Python 教程")
print(result.final_answer)

# 流式执行
for chunk in agent.stream("分析这个问题"):
    print(f"节点: {chunk['node']}, 状态: {chunk['state'].status}")
```

---

## 核心组件

### AgentState - 状态管理

```python
from backend.algo.langgraph import AgentState, AgentMessage

# 创建状态
state = AgentState(input="用户问题")

# 添加消息
state.add_message(AgentMessage.human("你好"))
state.add_message(AgentMessage.ai("你好！有什么可以帮助你的？"))

# 检查状态
print(state.iteration)          # 当前迭代
print(state.status)             # 执行状态
print(len(state.messages))      # 消息数量
print(state.final_answer)       # 最终答案

# 工具调用
state.add_tool_call(ToolCall(id="1", name="search", arguments={"query": "test"}))
```

### Tool - 工具定义

```python
from backend.algo.langgraph import tool, async_tool, Tool, ToolParameter

# 使用装饰器
@tool(name="calculator", description="计算器")
def calculator(expression: str) -> str:
    return str(eval(expression))

# 异步工具
@async_tool(name="async_search")
async def async_search(query: str) -> str:
    await asyncio.sleep(1)
    return f"结果: {query}"

# 手动创建
my_tool = Tool(
    name="custom_tool",
    description="自定义工具",
    func=lambda x: f"处理: {x}",
    parameters=[
        ToolParameter(name="x", type="str", description="输入参数")
    ]
)
```

### StateGraph - 状态图

```python
from backend.algo.langgraph import (
    StateGraph,
    GraphConfig,
    LLMNode,
    ToolNode,
    route_after_agent
)

# 创建图
graph = StateGraph(config=GraphConfig(name="my_graph", max_iterations=10))

# 添加节点
graph.add_node("agent", LLMNode(name="agent", system_prompt="你是助手"))
graph.add_node("tools", ToolNode(name="tools"))

# 设置边
graph.set_entry_point("agent")
graph.add_conditional_edges("agent", route_after_agent, {
    "tools": "tools",
    "__end__": "__end__"
})
graph.add_edge("tools", "agent")

# 编译并执行
compiled = graph.compile()
result = compiled.invoke("用户问题")
```

---

## Agent 类型

### ReActAgent

推理-行动循环，适用于需要工具调用的任务。

```python
from backend.algo.langgraph import create_react_agent

agent = create_react_agent(
    name="react_agent",
    tools=[search._tool, calculator._tool],
    system_prompt="你是一个智能助手",
    max_iterations=10,
    temperature=0.7
)

result = agent.invoke("计算 123 * 456 然后搜索结果的含义")
```

### PlanAndExecuteAgent

先制定计划，再逐步执行。

```python
from backend.algo.langgraph import create_plan_execute_agent

agent = create_plan_execute_agent(
    name="planner",
    tools=[search._tool],
    max_iterations=15,
    enable_replanning=True
)

result = agent.invoke("完成一个复杂的多步骤任务")
print(result.plan)  # 查看计划
```

### ReflexionAgent

通过自我反思改进输出。

```python
from backend.algo.langgraph import ReflexionAgent, AgentConfig

config = AgentConfig(name="reflexion", max_iterations=10)
agent = ReflexionAgent(
    config=config,
    tools=[],
    max_reflections=3
)

result = agent.invoke("写一篇关于 AI 的文章")
print(result.reflections)  # 查看反思过程
```

### MultiAgentSystem

多 Agent 协作系统。

```python
from backend.algo.langgraph import MultiAgentSystem, AgentRole, AgentConfig

roles = [
    AgentRole(
        name="researcher",
        description="负责信息搜索",
        system_prompt="你是研究员",
        tools=[search._tool]
    ),
    AgentRole(
        name="analyst",
        description="负责数据分析",
        system_prompt="你是分析师",
        tools=[calculator._tool]
    )
]

config = AgentConfig(name="team", max_iterations=20)
multi_agent = MultiAgentSystem(
    config=config,
    roles=roles,
    collaboration_mode="supervisor"  # 或 "round_robin"
)

result = multi_agent.invoke("分析市场数据并给出建议")
```

---

## 检查点

### 内存检查点

```python
from backend.algo.langgraph import MemoryCheckpointer

checkpointer = MemoryCheckpointer(max_checkpoints=100)

# 保存状态
checkpoint_id = checkpointer.save(state)

# 加载状态
loaded_state = checkpointer.load(checkpoint_id)

# 获取最新
latest = checkpointer.get_latest(thread_id="thread_123")
```

### Redis 检查点

```python
from backend.algo.langgraph import RedisCheckpointer

checkpointer = RedisCheckpointer(
    host="localhost",
    port=6379,
    prefix="langgraph:",
    ttl=86400 * 7  # 7天过期
)

# 在 Agent 中使用
agent = create_react_agent(
    name="agent",
    tools=[],
    checkpointer=checkpointer
)
```

---

## 内置工具

| 工具 | 类别 | 说明 |
|------|------|------|
| `web_search` | SEARCH | 搜索互联网 |
| `knowledge_search` | RETRIEVAL | 知识库搜索 |
| `python_executor` | CODE | 执行 Python 代码 |
| `sql_executor` | DATA | 执行 SQL 查询 |
| `data_analyzer` | DATA | 数据统计分析 |
| `data_transformer` | DATA | 数据转换 |
| `calculator` | COMPUTATION | 数学计算 |
| `current_time` | SYSTEM | 获取当前时间 |
| `json_parser` | DATA | JSON 解析 |
| `regex_matcher` | DATA | 正则匹配 |
| `http_request` | API | HTTP 请求 |

```python
from backend.algo.langgraph import get_builtin_tools, get_search_tools

# 获取所有内置工具
all_tools = get_builtin_tools()

# 获取特定类别
search_tools = get_search_tools()
```

---

## 高级用法

### 自定义节点

```python
from backend.algo.langgraph import BaseNode, NodeConfig, NodeType

class MyCustomNode(BaseNode):
    def __init__(self, name: str):
        config = NodeConfig(
            name=name,
            node_type=NodeType.CUSTOM,
            description="自定义节点"
        )
        super().__init__(config)
    
    def __call__(self, state: AgentState) -> AgentState:
        # 自定义逻辑
        state.add_message(AgentMessage.ai("自定义处理完成"))
        return state
```

### 条件边

```python
from backend.algo.langgraph import ConditionalEdge, EdgeCondition

def my_condition(state: AgentState) -> str:
    if state.pending_tool_calls:
        return "tools"
    if state.final_answer:
        return "__end__"
    return "agent"

graph.add_conditional_edges(
    "agent",
    my_condition,
    {
        "tools": "tools",
        "agent": "agent",
        "__end__": "__end__"
    }
)
```

### 人机协作

```python
from backend.algo.langgraph import HumanNode

human_node = HumanNode(
    name="human_review",
    prompt_template="请审核以下内容: {output}",
    timeout=300.0
)

graph.add_node("human", human_node)

# 执行时会暂停等待人工输入
for chunk in graph.stream(input_data):
    if chunk['state'].waiting_for_human:
        # 收集人工输入
        feedback = get_user_input()
        chunk['state'].receive_human_input(feedback)
```

---

## 与 LLM 集成

### OpenAI

```python
from openai import OpenAI

client = OpenAI(api_key="your-key")

agent = create_react_agent(
    name="openai_agent",
    tools=[search._tool],
    llm_client=client,
    model="gpt-4"
)
```

### Anthropic

```python
from anthropic import Anthropic

client = Anthropic(api_key="your-key")

agent = create_react_agent(
    name="claude_agent",
    tools=[search._tool],
    llm_client=client,
    model="claude-3-opus-20240229"
)
```

### 本地模型

```python
# 使用 Ollama 或其他本地模型
class LocalLLMClient:
    def chat_completions_create(self, **kwargs):
        # 调用本地模型
        pass

agent = create_react_agent(
    name="local_agent",
    tools=[],
    llm_client=LocalLLMClient()
)
```

---

## 错误处理

```python
try:
    result = agent.invoke("复杂任务")
except Exception as e:
    print(f"执行失败: {e}")
    
    # 检查中间状态
    state = agent._graph.get_state()
    if state:
        print(f"最后状态: {state.status}")
        print(f"错误信息: {state.error}")
```

---

## 性能优化

1. **合理设置 max_iterations**
   ```python
   agent = create_react_agent(max_iterations=5)  # 简单任务
   agent = create_react_agent(max_iterations=15) # 复杂任务
   ```

2. **使用工具缓存**
   ```python
   @tool(name="cached_search", timeout=30.0)
   @functools.lru_cache(maxsize=100)
   def cached_search(query: str) -> str:
       return perform_search(query)
   ```

3. **并行工具执行**
   ```python
   tool_node = ToolNode(name="tools", parallel=True)
   ```

---

## 许可证

Copyright © 2024 VectorSphere. All rights reserved.


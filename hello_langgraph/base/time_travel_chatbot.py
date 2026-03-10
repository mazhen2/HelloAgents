"""
LangGraph 时间旅行示例

本示例演示了 LangGraph 中的时间旅行功能，包括：
1. 构建带有检查点的图
2. 添加步骤并保存状态历史
3. 获取状态历史并回溯
4. 从某个检查点恢复执行

时间旅行允许用户回溯聊天机器人的工作流程，探索不同的结果，
或者在出错时回退到之前的状态重新执行。
"""

import os
from typing import Annotated

# 从 LangChain Tavily 集成导入搜索工具
# TavilySearch 是一个专为 LLM 设计的 AI 搜索引擎，能提供高质量的搜索结果
from langchain_tavily import TavilySearch

# 导入 TypedDict 用于定义结构化字典类型
# TypedDict 允许为字典指定键名和对应的值类型，提供编译时类型检查
from typing_extensions import TypedDict

# 导入 LangGraph 内存检查点保存器
# MemorySaver 提供内存中的状态持久化，支持对话线程的记忆功能
# 这是实现时间旅行功能的关键组件
from langgraph.checkpoint.memory import MemorySaver

# 导入 LangGraph 图构建的核心组件
# StateGraph: 基于状态的图结构，用于构建多步骤工作流
# START: 图的起始节点标识符
from langgraph.graph import StateGraph, START

# 导入消息聚合函数 add_messages
# 该函数定义了如何合并多条消息到状态中，确保消息列表正确累积而不是被覆盖
from langgraph.graph.message import add_messages

# 导入 LangGraph 预构建的工具处理组件
# ToolNode: 预定义的节点，专门用于执行工具调用
# tools_condition: 条件判断函数，决定是否需要调用工具
from langgraph.prebuilt import ToolNode, tools_condition

# 从本地模块导入已配置的 LLM
from hello_langgraph.base.chatbot import llm

# 设置 Tavily 搜索 API 密钥
os.environ["tavily_api_key"] = "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"


# =============================================================================
# 定义图的状态类型
# =============================================================================
class State(TypedDict):
    """
    图的状态定义
    
    在 LangGraph 中，State 是贯穿整个图执行过程的数据结构。
    每个节点都可以读取和更新状态，状态在节点之间传递。
    """
    # messages 字段存储完整的对话历史消息列表
    # Annotated[list, add_messages] 表示这是一个列表类型，使用 add_messages 函数进行合并
    # 当多个节点同时返回 messages 时，add_messages 会将它们聚合而不是覆盖
    messages: Annotated[list, add_messages]


# =============================================================================
# 构建图结构
# =============================================================================

# 创建 StateGraph 实例，传入之前定义的 State 类型
# graph_builder 是图的构建器对象，用于配置图的结构和行为
graph_builder = StateGraph(State)

# 创建 Tavily 搜索工具实例，限制最多返回 2 个搜索结果
# max_results 参数控制搜索的广度和成本，较少的结果意味着更快的响应
tool = TavilySearch(max_results=2)

# 构建可用工具列表，目前只包含搜索工具
tools = [tool]

# 将工具集合绑定到 LLM，创建一个支持工具调用的 LLM 实例
# bind_tools 方法让 LLM 能够识别可用工具并在适当时候生成工具调用请求
llm_with_tools = llm.bind_tools(tools)


# =============================================================================
# 定义节点函数
# =============================================================================

def chatbot(state: State):
    """
    聊天机器人节点函数
    
    这是图中的一个处理节点，负责与 LLM 进行单轮对话交互。
    该节点会分析对话历史，决定是否需要调用工具。
    
    参数:
        state: 包含当前图状态的对象，至少包含 messages 字段
        
    返回:
        dict: 包含新消息的状态更新字典
    """
    # 使用当前状态中的消息历史调用 LLM
    # invoke 方法会触发 LLM 的推理过程，可能生成普通回复或工具调用请求
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


# =============================================================================
# 添加节点到图中
# =============================================================================

# 向图中添加 "chatbot" 节点
# add_node 方法注册了一个处理单元，当图执行到这个节点时会调用 chatbot 函数
# 节点名称 "chatbot" 用于在后续的路由配置中引用这个节点
graph_builder.add_node("chatbot", chatbot)

# 创建 ToolNode 实例，专门用于处理工具执行
# ToolNode 是 LangGraph 预构建的组件，自动处理工具查找、参数解析和结果封装
tool_node = ToolNode(tools=[tool])

# 向图中添加 "tools" 节点，负责执行所有已绑定的工具
# 当路由决策指向工具执行时，图会进入这个节点
graph_builder.add_node("tools", tool_node)

# =============================================================================
# 添加边（连接节点）
# =============================================================================

# 添加条件边，实现从 chatbot 节点出发的动态路由
# add_conditional_edges 根据运行时条件决定下一步走向哪个节点
graph_builder.add_conditional_edges(
    "chatbot",  # 源节点名称，表示这些边从 "chatbot" 节点出发
    tools_condition,  # 条件函数，分析 chatbot 的输出并决定路由目标
    # tools_condition 会检查消息中是否包含工具调用请求：
    # - 如果有工具调用，返回 "tools" 节点名称
    # - 如果没有工具调用，返回 END 表示对话结束
)

# 添加从 tools 节点回到 chatbot 节点的有向边
# 这形成了一个循环：chatbot -> tools -> chatbot
# 循环的意义在于：工具执行完成后，需要回到 chatbot 让 LLM 基于工具结果继续对话
graph_builder.add_edge("tools", "chatbot")

# 添加从 START 到 chatbot 的起始边
# 这条边定义了图的入口点，当图开始执行时首先进入 chatbot 节点
graph_builder.add_edge(START, "chatbot")

# =============================================================================
# 编译图
# =============================================================================

# 创建内存检查点保存器实例
# memory 对象负责在每次状态变化后保存快照，支持断点续传和多线程对话
# 这是实现时间旅行功能的核心组件
memory = MemorySaver()

# 编译图构建器，生成可执行的图对象
# checkpointer 参数启用了状态持久化，使得图可以在中断后恢复
# compile 方法会验证图的结构完整性（如检查是否有孤立的节点）
graph = graph_builder.compile(checkpointer=memory)

from typing import Annotated

from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from hello_langgraph.base.chatbot import llm


# ==================== 定义状态结构 ====================

class State(TypedDict):
    """
    定义 LangGraph 的状态结构

    messages: 消息列表，存储对话历史
    add_messages: 消息合并函数，确保新消息正确追加到列表中
    """
    messages: Annotated[list, add_messages]


# ==================== 初始化图构建器 ====================

graph_builder = StateGraph(State)


# ==================== 定义工具函数 ====================

def get_weather(city: str) -> str:
    """
    获取城市天气信息（模拟工具）

    Args:
        city: 城市名称

    Returns:
        天气描述字符串
    """
    return f"{city} 今天天气晴朗，25°C"


def get_time() -> str:
    """
    获取当前时间

    Returns:
        格式化的时间字符串 (YYYY-MM-DD HH:MM:SS)
    """
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ==================== 绑定工具到 LLM ====================

# 将工具函数注册为列表，LLM 可以调用这些函数
tools = [get_weather, get_time]

# 将工具绑定到 LLM，使其具备工具调用能力
# llm_with_tools 现在可以识别何时需要调用工具，并返回工具调用请求
llm_with_tools = llm.bind_tools(tools)


# ==================== 定义聊天节点 ====================

def chatbot(state: State):
    """
    聊天机器人节点处理函数

    Args:
        state: 当前状态，包含消息列表

    Returns:
        包含 AI 回复消息的字典
    """
    # 使用绑定工具的 LLM 处理用户消息
    # LLM 会判断是否需要调用工具，如果需要则返回工具调用请求
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


# ==================== 构建图结构 ====================

# 添加聊天机器人节点到图中
graph_builder.add_node("chatbot", chatbot)

# 创建工具执行节点
# ToolNode 会自动执行 LLM 请求的工具调用
tool_node = ToolNode(tools=tools)
graph_builder.add_node("tools", tool_node)

# 添加条件边：根据 LLM 的输出决定下一步走向
# tools_condition 会检查消息中是否有工具调用请求
# - 如果有工具调用 → 路由到 "tools" 节点
# - 如果没有 → 直接结束对话
graph_builder.add_conditional_edges(
    "chatbot",  # 从 chatbot 节点出发
    tools_condition,  # 条件判断函数
)

# 添加工具执行后的回流边
# 当工具执行完毕后，返回 chatbot 节点继续对话
# 这样 LLM 可以根据工具返回的结果生成最终回复
graph_builder.add_edge("tools", "chatbot")

# 添加起始边：从 START 到 chatbot 节点
graph_builder.add_edge(START, "chatbot")

# ==================== 编译图 ====================

# 编译图，使其可以执行
graph = graph_builder.compile()

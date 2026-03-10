# 从 typing 模块导入 Annotated，用于给类型添加元数据和约束
from typing import Annotated

# 从 langchain.chat_models 导入聊天模型初始化函数
from langchain.chat_models import init_chat_model
# 从 langchain_tavily 导入 TavilySearch 搜索工具（虽然本例未直接使用，但可能是预留）
from langchain_tavily import TavilySearch
# 从 langchain_core.messages 导入基础消息类，用于类型提示
from langchain_core.messages import BaseMessage
# 从 typing_extensions 导入 TypedDict，用于定义带有类型提示的字典
from typing_extensions import TypedDict

# 从 langgraph.checkpoint.memory 导入 MemorySaver，用于保存和恢复图的状态（实现记忆功能）
from langgraph.checkpoint.memory import MemorySaver
# 从 langgraph.graph 导入 StateGraph，用于构建状态驱动的有向图
from langgraph.graph import StateGraph
# 从 langgraph.graph.message 导入 add_messages，这是一个reducer函数
# 用于在状态更新时将新消息追加到现有消息列表中
from langgraph.graph.message import add_messages
# 从 langgraph.prebuilt 导入预构建的工具节点和条件判断函数
# ToolNode: 执行工具调用的节点
# tools_condition: 判断LLM输出是否包含工具调用的条件函数
from langgraph.prebuilt import ToolNode, tools_condition

# 从同目录下的 add_tools_chatbot 模块导入工具函数
# get_weather: 获取天气信息的工具
# get_time: 获取当前时间的工具
from hello_langgraph.base.add_tools_chatbot import get_weather, get_time
# 从同目录下的 chatbot 模块导入已初始化的 LLM 实例
from hello_langgraph.base.chatbot import llm


# 定义状态类型，使用 TypedDict 确保类型安全
# State 类定义了图的全局状态结构
class State(TypedDict):
    # messages 字段是一个消息列表，Annotated 用于添加元数据
    # add_messages 是 reducer 函数：每次状态更新时，将新消息追加到列表而非替换
    # 这确保了对话历史会被保留和累积
    messages: Annotated[list, add_messages]


# 创建状态图构建器，指定状态类型为 State
# StateGraph 是 LangGraph 的核心组件，用于定义节点和边的流程
graph_builder = StateGraph(State)

# 定义可用的工具列表
# 这些工具函数会被绑定到 LLM，使其能够根据用户输入决定是否调用工具
# 当前包含两个工具：获取天气和获取时间
tools = [get_weather, get_time]

# 将工具绑定到 LLM，创建具有工具调用能力的 LLM 实例
# bind_tools 方法让 LLM 能够：
# 1. 识别用户问题是否需要调用工具
# 2. 生成符合工具参数规范的工具调用请求
# 返回的 llm_with_tools 可以在回复中包含工具调用指令
llm_with_tools = llm.bind_tools(tools)


# 定义 chatbot 节点的处理函数
# 该函数接收当前状态，调用 LLM 生成回复
# 参数 state: 包含 messages（对话历史）的状态字典
# 返回: 更新后的状态，包含 LLM 生成的新消息
def chatbot(state: State):
    # 调用绑定了工具的 LLM 处理消息历史
    # invoke 方法接收消息列表，返回 LLM 的回复（可能是文本或工具调用）
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


# 将 chatbot 函数注册为图中的一个节点，节点名为 "chatbot"
# 这个节点负责调用 LLM 生成对话内容
graph_builder.add_node("chatbot", chatbot)

# 创建工具节点，使用 LangGraph 预置的 ToolNode
# ToolNode 会执行 LLM 请求的工具调用，并返回工具的执行结果
# 参数 tools=[tools] 指定要执行的工具列表
tool_node = ToolNode(tools=[tools])
# 将工具节点添加到图中，节点名为 "tools"
graph_builder.add_node("tools", tool_node)


# 添加条件边：从 chatbot 节点出发
# tools_condition 是一个条件函数，会检查 LLM 的输出是否包含工具调用
# - 如果包含工具调用：边指向 "tools" 节点
# - 如果不包含工具调用（直接回复）：边指向 END（结束图）
graph_builder.add_conditional_edges(
    "chatbot",  # 源节点
    tools_condition,  # 条件函数
)

# 添加普通边：从 tools 节点回到 chatbot 节点
# 工具执行完毕后，需要将结果返回给 LLM，让它生成最终回复
graph_builder.add_edge("tools", "chatbot")

# 设置图的入口点为 chatbot 节点
# 这是图的起始节点，对话流程从这开始
graph_builder.set_entry_point("chatbot")

# 创建内存检查点保存器
# MemorySaver 将图的状态保存在内存中，实现跨对话的记忆功能
# 用户关闭应用后，只要进程未结束，后续对话都能访问之前的上下文
memory = MemorySaver()

# 编译图，生成可执行的图对象
# checkpointer 参数指定使用 memory 保存器来维护对话状态
# 编译后的 graph 对象可以接收输入并执行完整的对话流程
graph = graph_builder.compile(checkpointer=memory)

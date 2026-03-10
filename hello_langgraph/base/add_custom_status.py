# 导入类型注解模块，用于支持 Annotated 类型提示
# Annotated 允许在类型上附加元数据，这里用于标记特殊参数的注入方式
import os
from typing import Annotated

# 从 LangChain Tavily 集成包中导入搜索工具
# TavilySearch 是一个基于 AI 的搜索引擎，专为 LLM 设计，能提供高质量的搜索结果
from langchain_tavily import TavilySearch

# 导入 LangChain 核心消息类型 ToolMessage
# ToolMessage 用于表示工具执行的结果，会添加到对话历史中供模型参考
from langchain_core.messages import ToolMessage

# 导入 LangChain 工具系统的核心组件
# tool: 装饰器，将普通函数转换为 LLM 可调用的工具
# InjectedToolCallId: 特殊注解，标记不应该暴露给模型的参数（工具调用 ID）
from langchain_core.tools import InjectedToolCallId, tool

# 导入 TypedDict 用于定义结构化字典类型
# TypedDict 允许为字典指定键名和对应的值类型，提供编译时类型检查
from typing_extensions import TypedDict

# 导入 LangGraph 内存检查点保存器
# MemorySaver 提供内存中的状态持久化，支持对话线程的记忆功能
from langgraph.checkpoint.memory import MemorySaver

# 导入 LangGraph 图构建的核心组件
# StateGraph: 基于状态的图结构，用于构建多步骤工作流
# START: 图的起始节点标识符
# END: 图的结束节点标识符
from langgraph.graph import StateGraph, START, END

# 导入消息聚合函数 add_messages
# 该函数定义了如何合并多条消息到状态中，确保消息列表正确累积而不是被覆盖
from langgraph.graph.message import add_messages

# 导入 LangGraph 预构建的工具处理组件
# ToolNode: 预定义的节点，专门用于执行工具调用
# tools_condition: 条件判断函数，决定是否需要调用工具
from langgraph.prebuilt import ToolNode, tools_condition

# 导入 LangGraph 流程控制类型
# Command: 用于精确控制状态更新和执行流程的命令对象
# interrupt: 中断函数，暂停执行并等待人类输入
from langgraph.types import Command, interrupt

from hello_langgraph.base.chatbot import llm

os.environ["tavily_api_key"] = "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"
os.environ["LLM_MODEL_ID"] = "qwen3.5-flash"


# 定义图的状态类型，继承自 TypedDict
# State 类描述了图在执行过程中需要维护和传递的所有数据字段
class State(TypedDict):
    # messages 字段存储完整的对话历史消息列表
    # Annotated[list, add_messages] 表示这是一个列表类型，使用 add_messages 函数进行合并
    # 当多个节点同时返回 messages 时，add_messages 会将它们聚合而不是覆盖
    messages: Annotated[list, add_messages]

    # name 字段存储经过人工验证的名称信息
    # 初始为空字符串，在 human_assistance 工具执行后被填充
    name: str

    # birthday 字段存储经过人工验证的日期信息
    # 同样在 human_assistance 工具执行后被填充
    birthday: str


 # 定义人工协助工具函数
# @tool 装饰器会自动将函数注册为 LLM 可调用的工具
# 工具描述会自动从函数的 docstring 中提取
@tool
def human_assistance(
        # 待验证的名称参数，由 LLM 根据对话内容提供
        name: str,
        # 待验证的日期参数，由 LLM 根据查询结果提供
        birthday: str,
        # 工具调用 ID，使用 InjectedToolCallId 注解
        # 这个参数对模型隐藏，但框架会在执行时自动注入对应的工具调用标识符
        # ToolMessage 需要这个 ID 来关联请求和响应
        tool_call_id: Annotated[str, InjectedToolCallId]
) -> str:
    """向人工请求协助，核对并修正姓名和日期等关键信息。

    典型流程：
    1. LLM 先通过搜索工具得到一个候选答案（带有 name / birthday 等字段）。
    2. LLM 调用本工具 `human_assistance`，把候选的 name / birthday 传进来。
    3. 本工具通过 `interrupt(...)` 暂停图的执行，把需要人工确认的数据抛给外部（例如 CLI / Web UI）。
    4. 外部人类在界面上确认或修改这些字段，然后通过 `resume(data=...)` 把结果传回图中。
    5. 本工具根据人类给的数据更新状态里的 `name` 和 `birthday` 字段，并把一条 `ToolMessage` 写回对话历史。
    """
    # 调用 interrupt 函数暂停工作流执行，进入人机交互模式。
    # interrupt 会向 UI 或前端发送一个包含问题和上下文的信号。
    # 执行流在这里阻塞，直到人类通过 resume 提供响应。
    human_response = interrupt(
        {
            # 向人类展示的问题文本（中文描述更直观）
            "question": "下面的姓名和日期是否正确？如有需要，请修改后再确认。",
            # 待确认的名称值
            "name": name,
            # 待确认的日期值
            "birthday": birthday,
        },
    )

    # 检查人类的响应是否确认信息正确。
    # 从响应字典中获取 "correct" 字段，转为小写后检查是否以 'y' 开头，
    # 这样可以匹配 "yes", "Yes", "YES", "yep" 等多种肯定表达。
    if human_response.get("correct", "").lower().startswith("y"):
        # 如果人类确认正确，直接使用原始提供的值作为已验证信息。
        verified_name = name
        verified_birthday = birthday
        response = "Correct"  # 设置简洁的确认响应消息

    # 如果人类指出信息有误并提供修正。
    else:
        # 从人类响应中获取修正后的值，使用 get 方法提供默认值以防字段缺失。
        # 如果人类没有提供某个字段的修正，则保持原值不变。
        verified_name = human_response.get("name", name)
        verified_birthday = human_response.get("birthday", birthday)
        # 记录进行了人工修正（包括人类传回的完整 payload，便于追踪）。
        response = f"Made a correction: {human_response}"

    # 构建状态更新字典，明确指定要修改的状态字段。
    state_update = {
        # 更新状态中的名称为已验证的值
        "name": verified_name,
        # 更新状态中的日期为已验证的值
        "birthday": verified_birthday,
        # 添加工具响应消息到对话历史。
        # ToolMessage 将响应内容与特定的工具调用 ID 关联。
        "messages": [ToolMessage(response, tool_call_id=tool_call_id)],
    }

    # 返回 Command 对象来控制状态更新。
    # Command 是 LangGraph 提供的强大机制，允许在工具内部精确控制状态变化，
    # update 参数指定要合并到图状态中的字段，而不是完全替换状态。
    return Command(update=state_update)


# 创建 Tavily 搜索工具实例，限制最多返回 2 个搜索结果
# max_results 参数控制搜索的广度和成本，较少的结果意味着更快的响应
tool = TavilySearch(max_results=2)

# 构建可用工具列表，包含搜索工具和人工协助工具
# 这两个工具都会被绑定到 LLM，模型可以根据需要选择调用
tools = [tool, human_assistance]

# 将工具集合绑定到 LLM，创建一个支持工具调用的 LLM 实例
# bind_tools 方法让 LLM 能够识别可用工具并在适当时候生成工具调用请求
llm_with_tools = llm.bind_tools(tools)


# 定义聊天机器人节点函数
# 这是图中的一个处理节点，负责与 LLM 进行单轮对话交互
def chatbot(state: State):
    # 使用当前状态中的消息历史调用 LLM
    # invoke 方法会触发 LLM 的推理过程，可能生成普通回复或工具调用请求
    message = llm_with_tools.invoke(state["messages"])

    # 断言检查：确保单次迭代中最多只有一个工具调用
    # 这是一个安全措施，防止复杂的工具调用链导致不可控的行为
    # 如果需要多个工具调用，会通过多次循环迭代来完成
    assert (len(message.tool_calls) <= 1)

    # 返回包含新消息的状态更新字典
    # LangGraph 会自动将这个更新合并到主状态中
    return {"messages": [message]}


# 创建 StateGraph 实例，传入之前定义的 State 类型
# graph_builder 是图的构建器对象，用于配置图的结构和行为
graph_builder = StateGraph(State)

# 向图中添加 "chatbot" 节点
# add_node 方法注册了一个处理单元，当图执行到这个节点时会调用 chatbot 函数
# 节点名称 "chatbot" 用于在后续的路由配置中引用这个节点
graph_builder.add_node("chatbot", chatbot)

# 创建 ToolNode 实例，专门用于处理工具执行
# ToolNode 是 LangGraph 预构建的组件，自动处理工具查找、参数解析和结果封装
tool_node = ToolNode(tools=tools)

# 向图中添加 "tools" 节点，负责执行所有已绑定的工具
# 当路由决策指向工具执行时，图会进入这个节点
graph_builder.add_node("tools", tool_node)

# 添加条件边，实现从 chatbot 节点出发的动态路由
# add_conditional_edges 根据运行时条件决定下一步走向哪个节点
graph_builder.add_conditional_edges(
    # 源节点名称，表示这些边从 "chatbot" 节点出发
    "chatbot",
    # 条件函数，分析 chatbot 的输出并决定路由目标
    # tools_condition 会检查消息中是否包含工具调用请求：
    # - 如果有工具调用，返回 "tools" 节点名称
    # - 如果没有工具调用，返回 END 表示对话结束
    tools_condition,
)

# 添加从 tools 节点回到 chatbot 节点的有向边
# 这形成了一个循环：chatbot -> tools -> chatbot
# 循环的意义在于：工具执行完成后，需要回到 chatbot 让 LLM 基于工具结果继续对话
graph_builder.add_edge("tools", "chatbot")

# 添加从 START 到 chatbot 的起始边
# 这条边定义了图的入口点，当图开始执行时首先进入 chatbot 节点
graph_builder.add_edge(START, "chatbot")

# 创建内存检查点保存器实例
# memory 对象负责在每次状态变化后保存快照，支持断点续传和多线程对话
memory = MemorySaver()

# 编译图构建器，生成可执行的图对象
# checkpointer 参数启用了状态持久化，使得图可以在中断后恢复
# compile 方法会验证图的结构完整性（如检查是否有孤立的节点）
graph = graph_builder.compile(checkpointer=memory)

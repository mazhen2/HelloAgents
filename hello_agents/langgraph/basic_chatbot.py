"""
基本聊天机器人示例
===================

本教程将构建一个基本聊天机器人，这是后续系列教程的基础。
在这个教程中，您将逐步了解 LangGraph 的关键概念。

先决条件:
    - 支持工具调用功能的 LLM (OpenAI, Anthropic, 阿里云通义千问 或 Google Gemini)
    - 已安装所需包: pip install -U langgraph langsmith langchain

注册 LangSmith 可快速发现问题并提高 LangGraph 项目的性能。
有关如何开始的更多信息，请参阅 LangSmith 文档。

使用说明:
    1. 确保 .env 文件中配置了 LLM_API_KEY, LLM_MODEL_ID, LLM_BASE_URL
    2. 运行本脚本: python basic_chatbot.py
    3. 输入您的问题，与聊天机器人对话
    4. 输入 quit, exit 或 q 退出聊天
"""

from typing import Annotated

from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages


# =============================================================================
# 步骤 1: 定义状态 (State)
# =============================================================================
# 状态包括图的模式和处理状态更新的 reducer 函数
# 状态是一个具有一个键: messages 的 TypedDict
# add_messages reducer 函数用于将新消息追加到列表中，而不是覆盖它


class State(TypedDict):
    """
    聊天机器人的状态定义

    Attributes:
        messages: 消息列表，存储对话历史
    """
    # 消息的类型是“list”。add_messages函数
    # 在注释中定义该状态键应该如何更新
    # （在本例中，它将消息追加到列表中，而不是覆盖它们）
    messages: Annotated[list, add_messages]


# =============================================================================
# 步骤 2: 创建 StateGraph
# =============================================================================
# StateGraph 对象将聊天机器人结构定义为"状态机"
# 我们将添加节点来表示 LLM 和聊天机器人可以调用的函数
# 并添加边来指定机器人应如何在这些函数之间进行转换


graph_builder = StateGraph(State)
"""
图构建器:
    - 每个节点都可以接收当前状态作为输入，并输出状态的更新
    - 对消息的更新将追加到现有列表而不是覆盖它
    - 这得益于与 Annotated 语法一起使用的预构建 add_messages 函数
"""

# =============================================================================
# 步骤 3: 初始化聊天模型
# =============================================================================
# 使用阿里云通义千问模型 (DashScope)
# 配置从 .env 文件或环境变量读取


import os

# 阿里云 DashScope 配置
LLM_MODEL_ID = os.getenv("LLM_MODEL_ID", "qwen-plus")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "60"))

# 初始化通义千问模型
# DashScope 的 API 兼容 OpenAI 格式，所以使用 ChatOpenAI
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model=LLM_MODEL_ID,
    api_key=LLM_API_KEY,
    base_url=LLM_BASE_URL,
    timeout=LLM_TIMEOUT,
    temperature=0.7,
)

# =============================================================================
# 步骤 4: 定义 chatbot 节点
# =============================================================================
# 节点表示工作单元，通常是普通的 Python 函数


def chatbot(state: State) -> dict:
    """
    Chatbot 节点函数
    
    该函数:
    - 接收当前状态作为输入
    - 返回包含更新的消息列表的字典
    
    这是所有 LangGraph 节点函数的基本模式。
    状态中的 add_messages 函数会将 LLM 的响应消息追加到状态中已有的消息之后。
    
    Args:
        state: 当前状态，包含 messages 列表
        
    Returns:
        dict: 包含新消息的字典
    """
    return {"messages": [llm.invoke(state["messages"])]}


# 第一个参数是唯一的节点名
# 第二个参数是无论何时都会被调用的函数或对象
#节点已被使用。
graph_builder.add_node("chatbot", chatbot)

# =============================================================================
# 步骤 5: 添加入口点
# =============================================================================
# 入口点告诉图每次运行时从何处开始工作


graph_builder.add_edge(START, "chatbot")

# =============================================================================
# 步骤 6: 编译图
# =============================================================================
# 在运行图之前，需要对其进行编译
# 调用 compile() 将创建一个 CompiledGraph，可以在状态上调用它


graph = graph_builder.compile()


# =============================================================================
# 步骤 7: 可视化图 (可选)
# =============================================================================
# 使用 get_graph 方法和其中一个"绘图"方法可视化图
# 这些 draw 方法都需要额外的依赖项


def visualize_graph():
    """可视化图结构 (可选功能)"""
    try:
        # 尝试使用 Mermaid 格式
        print("\n=== 图结构 (Mermaid) ===")
        print(graph.get_graph().draw_mermaid())
    except Exception:
        pass

    try:
        # 尝试使用 ASCII 格式
        print("\n=== 图结构 (ASCII) ===")
        print(graph.get_graph().draw_ascii())
    except Exception:
        pass


# =============================================================================
# 步骤 8: 运行聊天机器人
# =============================================================================


def stream_graph_updates(user_input: str):
    """
    流式处理图更新并打印响应
    
    这个函数负责:
    1. 将用户输入包装成消息格式，发送给 LangGraph 图
    2. 接收图的流式输出（每个节点的执行结果）
    3. 提取并打印 Assistant 的回复
    
    Args:
        user_input: 用户输入的文本
    """
    # graph.stream() 是 LangGraph 的核心方法之一
    # 它以"流式"方式执行图，并Yield每个节点执行的结果
    # 
    # 输入格式: {"messages": [...]} 
    #   - messages 是一个消息列表，每条消息包含 role 和 content
    #   - role 可以是 "user"(用户)、"assistant"(AI)、"system"(系统)
    #
    # 输出: 一个生成器，每次Yield一个节点的结果
    #   节点结果是一个字典，键是节点名，值是该节点的输出状态
    for event in graph.stream({"messages": [{"role": "user", "content": user_input}]}):
        # event 的结构示例: {"chatbot": {"messages": [AIMessage(content="...")]}}
        # 其中键是节点名称（如 "chatbot"），值是该节点的输出状态
        
        # 遍历事件中的所有值（通常是单个节点的结果）
        for value in event.values():
            # value 的结构: {"messages": [AIMessage(content="...")]}
            # messages 是一个包含所有消息的列表，包括用户输入和AI回复
            # 
            # [-1] 表示取列表中最后一条消息，即最新的AI回复
            # 打印格式: "Assistant: [AI的回复内容]"
            print("Assistant:", value["messages"][-1].content)


def run_chatbot():
    """
    运行聊天机器人的主循环
    """
    print("=" * 50)
    print("欢迎使用 LangGraph 基本聊天机器人!")
    print("输入您的问题，或输入 quit/exit/q 退出")
    print("=" * 50)

    while True:
        try:
            user_input = input("\nUser: ")
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break
            stream_graph_updates(user_input)
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            # fallback if input() is not available
            print(f"\n输入出错: {e}")
            user_input = "What do you know about LangGraph?"
            print(f"User: {user_input}")
            stream_graph_updates(user_input)
            break


# =============================================================================
# 主程序入口
# =============================================================================


if __name__ == "__main__":
    # 可选: 可视化图结构
    # visualize_graph()

    # 运行聊天机器人
    run_chatbot()

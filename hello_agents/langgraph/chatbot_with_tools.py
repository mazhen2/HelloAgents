"""
带工具的聊天机器人示例
=======================

本教程在基本聊天机器人的基础上，添加了网页搜索工具。
聊天机器人可以使用此工具查找相关信息并提供更好的回复。

先决条件:
    - Tavily 搜索引擎的 API 密钥 (https://tavily.com)
    - 或使用其他支持的搜索工具

使用说明:
    1. 设置 TAVILY_API_KEY 环境变量
    2. 运行本脚本: python chatbot_with_tools.py
    3. 输入您的问题，与聊天机器人对话
    4. 聊天机器人会自动判断是否需要使用搜索工具
    5. 输入 quit, exit 或 q 退出聊天
"""

from typing import Annotated

from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages


# =============================================================================
# 步骤 1: 定义状态 (State)
# =============================================================================
# 与基本聊天机器人相同，状态包含消息列表


class State(TypedDict):
    """
    聊天机器人的状态定义
    
    Attributes:
        messages: 消息列表，存储对话历史（包括用户消息、AI消息和工具消息）
    """
    # 消息的类型是"list"。add_messages函数
    # 在注释中定义该状态键应该如何更新
    # （在本例中，它将消息追加到列表中，而不是覆盖它们）
    messages: Annotated[list, add_messages]


# =============================================================================
# 步骤 2: 创建 StateGraph
# =============================================================================


graph_builder = StateGraph(State)


# =============================================================================
# 步骤 3: 初始化聊天模型和工具
# =============================================================================
# 使用阿里云通义千问模型 (DashScope)
# 并集成 Tavily 搜索引擎


import os


# 阿里云 DashScope 配置
LLM_MODEL_ID = os.getenv("LLM_MODEL_ID", "qwen-plus")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "60"))

# 初始化通义千问模型
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model=LLM_MODEL_ID,
    api_key=LLM_API_KEY,
    base_url=LLM_BASE_URL,
    timeout=LLM_TIMEOUT,
    temperature=0.7,
)

print(f"✓ 已初始化模型: {LLM_MODEL_ID}")
print(f"✓ API 地址: {LLM_BASE_URL}")


# =============================================================================
# 步骤 4: 配置搜索引擎工具
# =============================================================================
# 安装: pip install -U langchain-tavily
# 设置: export TAVILY_API_KEY=your-api-key


def _set_env(var: str):
    """从环境变量获取 API Key 并设置"""
    from os import environ
    if api_key := environ.get(var):
        return api_key
    raise EnvironmentError(f"未设置环境变量: {var}")


# 尝试获取 Tavily API Key
try:
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
    if TAVILY_API_KEY:
        from langchain_tavily import TavilySearch
        
        # 创建搜索工具，限制返回结果数量
        # 注意：新版本 langchain-tavily 需要显式传递 api_key 参数
        tool = TavilySearch(
            max_results=2,
            tavily_api_key=TAVILY_API_KEY  # 显式传递 API Key
        )
        tools = [tool]
        print("✓ 已加载 Tavily 搜索引擎")
    else:
        print("⚠ 未设置 TAVILY_API_KEY，将使用模拟工具")
        tools = []
except ImportError:
    print("⚠ 未安装 langchain-tavily，将使用模拟工具")
    tools = []


# =============================================================================
# 步骤 5: 绑定工具到 LLM
# =============================================================================
# 关键步骤：告诉 LLM 它可以调用哪些工具
# 通过 bind_tools 方法，LLM 知道如果需要使用工具，应该生成正确格式的 JSON


# 如果有可用的工具，则绑定到 LLM
if tools:
    # bind_tools 让 LLM 知道它可以调用哪些工具
    # 当 LLM 判断需要使用工具时，会返回包含 tool_calls 的消息
    llm_with_tools = llm.bind_tools(tools)
else:
    # 如果没有工具，使用普通 LLM
    llm_with_tools = llm


# =============================================================================
# 步骤 6: 定义节点
# =============================================================================
# 两个主要节点：
# 1. chatbot: 负责与 LLM 交互，决定是否需要调用工具
# 2. tools: 执行实际的工具调用


def chatbot(state: State):
    """
    Chatbot 节点函数
    
    该节点:
    - 接收当前状态（包含对话历史）
    - 调用 LLM（已绑定工具）生成回复
    - 如果 LLM 决定调用工具，会返回包含 tool_calls 的消息
    
    Args:
        state: 当前状态，包含 messages 列表
        
    Returns:
        dict: 包含新消息的字典
    """
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


graph_builder.add_node("chatbot", chatbot)


# =============================================================================
# 步骤 7: 创建工具节点
# =============================================================================
# 工具节点负责执行被调用的工具
# 这里提供两种实现方式：
# 1. BasicToolNode: 自己实现的简单版本
# 2. ToolNode: LangGraph 预构建的版本（推荐）


# 方式一：自己实现 BasicToolNode
# 适用于学习目的，了解工具调用的内部原理
import json
from langchain_core.messages import ToolMessage


class BasicToolNode:
    """
    基础工具节点
    
    负责执行 LLM 请求的工具调用。
    它会:
    1. 检查最后一条 AI 消息是否包含 tool_calls
    2. 对每个 tool_call，调用相应的工具
    3. 将工具执行结果包装成 ToolMessage 返回
    
    Attributes:
        tools_by_name: 工具名称到工具对象的映射字典
    """
    
    def __init__(self, tools: list) -> None:
        # 创建工具名称到工具对象的映射，方便根据名称查找
        self.tools_by_name = {tool.name: tool for tool in tools}
    
    def __call__(self, inputs: dict):
        """
        执行工具调用
        
        Args:
            inputs: 输入状态字典，必须包含 messages 键
            
        Returns:
            dict: 包含工具执行结果的字典
        """
        # 获取消息列表
        if messages := inputs.get("messages", []):
            # 取最后一条消息（通常是 AI 的回复，包含 tool_calls）
            message = messages[-1]
        else:
            raise ValueError("No message found in input")
        
        outputs = []
        
        # 遍历所有工具调用请求
        for tool_call in message.tool_calls:
            # 根据工具名称获取对应的工具对象
            tool = self.tools_by_name[tool_call["name"]]
            
            # 调用工具并获取结果
            # tool_call["args"] 包含工具调用所需的参数
            tool_result = tool.invoke(tool_call["args"])
            
            # 将工具结果包装成 ToolMessage
            # - content: 工具返回的内容（序列化为 JSON 字符串）
            # - name: 工具名称
            # - tool_call_id: 工具调用 ID，用于关联到原始请求
            outputs.append(
                ToolMessage(
                    content=json.dumps(tool_result),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )
        
        return {"messages": outputs}


# 方式二：使用 LangGraph 预构建的 ToolNode（更简洁）
# 推荐在实际项目中使用
try:
    from langgraph.prebuilt import ToolNode
    
    # 如果有工具，创建 ToolNode
    if tools:
        tool_node = ToolNode(tools=[tool])
        print("✓ 使用预构建 ToolNode")
    else:
        tool_node = None
except ImportError:
    # 如果没有预构建模块，使用自定义的 BasicToolNode
    if tools:
        tool_node = BasicToolNode(tools=[tools])
        print("✓ 使用自定义 BasicToolNode")
    else:
        tool_node = None


# 将工具节点添加到图中
# 注意：如果没有可用的工具，这一步会被跳过
if tool_node is not None:
    graph_builder.add_node("tools", tool_node)


# =============================================================================
# 步骤 8: 定义条件边 (Conditional Edges)
# =============================================================================
# 条件边根据当前状态决定下一步流向哪个节点
# 
# 路由逻辑:
# - 如果 AI 消息包含 tool_calls → 路由到 "tools" 节点执行工具
# - 如果 AI 消息不包含 tool_calls → 路由到 END，结束对话


def route_tools(
    state: State,
):
    """
    工具路由函数
    
    根据 AI 消息是否包含工具调用，决定下一步的流向。
    这是条件边的核心逻辑。
    
    Args:
        state: 当前状态
        
    Returns:
        str: "tools" 表示需要调用工具，END 表示直接回复用户
    """
    # 处理不同类型的状态输入
    if isinstance(state, list):
        # 如果状态是列表（某些情况下可能是这样）
        ai_message = state[-1]
    elif messages := state.get("messages", []):
        # 从状态字典中获取消息列表，取最后一条
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")
    
    # 检查 AI 消息是否有 tool_calls 属性且不为空
    # tool_calls 是 LLM 返回的工具调用请求列表
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    return END


# 添加条件边
# 第一个参数: 源节点（从哪个节点开始）
# 第二个参数: 路由函数（决定下一步去哪里）
# 第三个参数: 路由结果的映射（将路由函数的返回值映射到目标节点）
graph_builder.add_conditional_edges(
    "chatbot",           # 从 chatbot 节点开始
    route_tools,         # 使用 route_tools 函数决定下一步
    {
        "tools": "tools",  # 如果返回 "tools"，路由到 tools 节点
        END: END,          # 如果返回 END，结束对话
    },
)


# =============================================================================
# 步骤 9: 添加普通边 (Edges)
# =============================================================================
# 普通边指定固定的流向，不进行任何判断


# 工具执行完后，返回到 chatbot 节点，让 LLM 根据工具结果生成最终回复
# 这是 agent loop（代理循环）的关键：LLM → 工具 → LLM → ...
if tool_node is not None:
    graph_builder.add_edge("tools", "chatbot")


# 入口点：每次对话从 chatbot 节点开始
graph_builder.add_edge(START, "chatbot")


# =============================================================================
# 步骤 10: 编译图
# =============================================================================


graph = graph_builder.compile()


# =============================================================================
# 步骤 11: 可视化图 (可选)
# =============================================================================


def visualize_graph():
    """可视化图结构"""
    try:
        print("\n=== 图结构 (Mermaid) ===")
        print(graph.get_graph().draw_mermaid())
    except Exception:
        pass
    
    try:
        print("\n=== 图结构 (ASCII) ===")
        print(graph.get_graph().draw_ascii())
    except Exception:
        pass


# =============================================================================
# 步骤 12: 运行聊天机器人
# =============================================================================


def stream_graph_updates(user_input: str):
    """
    流式处理图更新并打印响应
    
    Args:
        user_input: 用户输入的文本
    """
    # graph.stream() 以流式方式执行图
    # 每次返回一个节点执行的结果
    for event in graph.stream({"messages": [{"role": "user", "content": user_input}]}):
        # 遍历事件中的所有值
        for value in event.values():
            # 获取最后一条消息（最新的 AI 回复或工具结果）
            message = value["messages"][-1]
            
            # 判断消息类型
            if hasattr(message, "tool_calls") and message.tool_calls:
                # 这是 AI 决定调用工具的响应
                # 打印工具调用信息（但不执行，因为工具节点会处理）
                print(f"Assistant: (正在调用工具...)")
            elif hasattr(message, "type") and message.type == "tool":
                # 这是工具执行的结果
                # ToolMessage 的 content 是 JSON 字符串格式的工具结果
                print(f"[工具结果]: {message.content[:200]}...")
            else:
                # 这是 AI 的最终回复
                print("Assistant:", message.content)


def run_chatbot():
    """
    运行聊天机器人的主循环
    """
    print("=" * 50)
    print("欢迎使用 LangGraph 带工具的聊天机器人!")
    print("我可以回答问题，必要时会自动搜索网页获取最新信息")
    print("输入 quit/exit/q 退出")
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
            print(f"\n出错: {e}")
            user_input = "What is LangGraph?"
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

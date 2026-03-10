# 导入类型注解相关模块
import os
from typing import Annotated

# 导入Tavily搜索工具 - 用于获取实时搜索结果
from langchain_tavily import TavilySearch
# 导入LangChain的工具装饰器 - 用于将函数转换为可调用的工具
from langchain_core.tools import tool
# 导入TypedDict - 用于定义带有类型提示的字典
from typing_extensions import TypedDict

# 导入LangGraph的状态检查点(记忆)相关模块
from langgraph.checkpoint.memory import MemorySaver
# 导入LangGraph核心组件: 状态图、起始节点、结束节点
from langgraph.graph import StateGraph, START, END
# 导入消息处理函数 - 用于在状态中累加消息列表
from langgraph.graph.message import add_messages
# 导入预定义的工具节点和工具条件判断函数
from langgraph.prebuilt import ToolNode, tools_condition
# 导入LangGraph类型: Command(命令)和interrupt(中断/人工介入)
from langgraph.types import Command, interrupt

# 从本地模块导入已配置好的大语言模型
from hello_langgraph.base.chatbot import llm

os.environ["tavily_api_key"] = "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"


# 定义图的状态类型 - 使用TypedDict确保类型安全
class State(TypedDict):
    """
    状态类型定义
    
    State字典将存储对话历史消息列表。
    使用add_messages注解可以在状态更新时自动将新消息追加到现有消息列表中，
    而不是完全替换整个列表。
    """
    # messages字段是一个注解为list类型的字段，使用add_messages作为reducer函数
    # 这意味着每次更新messages时，新消息会被追加到现有消息列表
    messages: Annotated[list, add_messages]


# 创建状态图构建器 - 用于定义整个工作流的结构
# 使用前面定义的State类型来规范状态的结构
graph_builder = StateGraph(State)


# 使用@tool装饰器将函数转换为LangChain工具
# 这个工具用于在需要人工帮助时暂停图执行并请求人工输入
@tool
def human_assistance(query: str) -> str:
    """
    请求人工协助的工具函数
    
    当AI无法独立完成任务时，可以通过调用此工具来请求人类帮助。
    该函数会中断图的执行流程，等待人工输入后再继续。
    
    参数:
        query: 需要人工帮助的问题或请求描述
        
    返回:
        人工提供的响应内容
    """
    # interrupt()是LangGraph中实现"人在环中"的关键函数
    # 它会暂停图的执行，并将传入的字典作为参数传递给外部调用者
    # 外部系统(如UI)可以捕获这个中断，获取query内容，然后提供人工输入
    # 返回的human_response字典中包含人工提供的数据
    human_response = interrupt({"query": query})
    # 从中断响应中提取人工提供的数据并返回
    return human_response["data"]


# 创建Tavily搜索工具实例
# max_results=2 指定每次搜索最多返回2条结果
tool = TavilySearch(max_results=2)

# 将所有可用的工具组合成列表
# 包括: 1) Tavily搜索工具  2) 人工协助工具
tools = [tool, human_assistance]

# 将工具绑定到LLM - 使大语言模型能够调用这些工具
# 绑定后，LLM会根据对话内容决定是否需要调用工具以及调用哪个工具
llm_with_tools = llm.bind_tools(tools)


# 定义聊天机器人节点的处理函数
# 每个节点函数接收当前状态作为输入，返回要更新到状态的值
def chatbot(state: State):
    """
    聊天机器人核心节点函数
    
    负责:
    1. 接收当前对话历史(包含用户消息和之前的AI回复)
    2. 调用绑定了工具的LLM生成回复
    3. LLM可能会选择调用工具(搜索或请求人工帮助)
    
    参数:
        state: 包含messages列表的当前状态
        
    返回:
        包含新生成消息的字典，将被合并到状态中
    """
    # 使用绑定了工具的LLM处理消息历史并生成回复
    # state["messages"]包含了之前的对话记录
    message = llm_with_tools.invoke(state["messages"])

    # 断言确保tool_calls数量不超过1个
    # 这是为了简化流程管理，确保每次只处理一个工具调用
    # (LangGraph的ToolNode设计为处理单个工具调用)
    assert (len(message.tool_calls) <= 1)

    # 将生成的回复消息返回，将被添加到状态中的messages列表
    return {"messages": [message]}


# 将chatbot函数添加为图中的一个节点
# 节点是图中的处理单元，每个节点执行特定的任务
graph_builder.add_node("chatbot", chatbot)

# 创建工具节点 - 这是一个预定义的节点
# ToolNode会自动处理工具调用，执行实际的工具并返回结果
# 它接收LLM生成的消息，检查是否有tool_calls，然后执行相应的工具
tool_node = ToolNode(tools=tools)

# 将工具节点添加到图中
graph_builder.add_node("tools", tool_node)

# 添加条件边 - 根据chatbot节点的输出决定后续流向
# tools_condition是LangGraph预定义的函数，用于判断消息是否包含工具调用
# 如果LLM决定调用工具，则流向"tools"节点
# 否则，流程结束(因为没有定义else分支，默认流向END)
graph_builder.add_conditional_edges(
    "chatbot",  # 源节点
    tools_condition,  # 条件函数 - 判断是否需要调用工具
)

# 添加从tools节点回到chatbot节点的边
# 工具执行完成后，将结果返回给LLM，让它基于工具结果生成最终回复
# 这样形成一个循环: chatbot -> tools -> chatbot -> ... 直到不需要更多工具调用
graph_builder.add_edge("tools", "chatbot")

# 添加从START到chatbot的边 - 定义图的入口点
# 工作流从chatbot节点开始
graph_builder.add_edge(START, "chatbot")

# 创建内存检查点存储 - 用于保存图执行的状态
# MemorySaver将状态保存在内存中，允许图在后续调用中恢复上下文
# 这对于多轮对话非常重要，可以记住之前的对话内容
memory = MemorySaver()

# 编译图 - 将所有节点、边和配置组合成可执行的图对象
# 编译后的graph可以接收输入并执行完整的工作流
graph = graph_builder.compile(checkpointer=memory)


def run_chatbot_with_human_in_loop():
    """
    运行带有"人在环中"功能的聊天机器人

    关键流程:
    1. 用户输入消息
    2. 图执行，当遇到 interrupt 时会暂停
    3. 从控制台读取人工输入
    4. 使用 Command(resume=...) 恢复执行
    """
    print("=" * 50)
    print("聊天机器人 (带人工介入功能)")
    print("输入 'quit' 或 'exit' 退出")
    print("=" * 50)

    # 配置 - 使用 thread_id 来保持对话状态
    config = {"configurable": {"thread_id": "chatbot-human-loop"}}

    while True:
        try:
            # 获取用户输入
            user_input = input("\n你: ").strip()

            if user_input.lower() in ["quit", "exit", "退出"]:
                print("再见!")
                break

            if not user_input:
                continue

            # 使用 stream 模式处理对话，可以捕获中断
            print("\nAI: ", end="", flush=True)

            try:
                # 流式输出
                for event in graph.stream(
                        {"messages": [("user", user_input)]},
                        config,
                        stream_mode="values"
                ):
                    if "messages" in event and event["messages"]:
                        # 获取最后一条消息
                        last_msg = event["messages"][-1]
                        # 如果是 AI 消息且有内容，打印出来
                        if hasattr(last_msg, "content") and last_msg.content:
                            print(last_msg.content, end="", flush=True)
            except Exception as e:
                # 检查是否是中断异常
                error_str = str(e)
                if "interrupt" in error_str.lower() or "Human" in error_str:
                    # 获取中断前的状态
                    checkpoint = graph.get_state(config)
                    print(f"\n\n[人工介入请求]")

                    # 尝试从 tool_calls 中获取 query
                    query = "请提供帮助"
                    if checkpoint and checkpoint.values.get("messages"):
                        messages = checkpoint.values["messages"]
                        for msg in reversed(messages):
                            if hasattr(msg, "tool_calls") and msg.tool_calls:
                                for tc in msg.tool_calls:
                                    if tc.get("name") == "human_assistance":
                                        args = tc.get("args", {})
                                        query = args.get("query", query)
                                        break

                    print(f"AI 请求人工帮助: {query}")
                    print("\n请提供您的回复 (输入后按回车):")

                    # 从控制台读取人工输入
                    human_input = input("> ").strip()

                    # 使用 Command 恢复执行
                    resume_config = {
                        "configurable": {"thread_id": "chatbot-human-loop"},
                        "resume": {"data": human_input}
                    }

                    # 继续执行
                    print("\n[继续执行...]\nAI: ", end="", flush=True)
                    for event in graph.stream(None, resume_config, stream_mode="values"):
                        if "messages" in event and event["messages"]:
                            last_msg = event["messages"][-1]
                            if hasattr(last_msg, "content") and last_msg.content:
                                print(last_msg.content, end="", flush=True)
                else:
                    print(f"\n错误: {e}")

            print()  # 换行

        except KeyboardInterrupt:
            print("\n\n退出...")
            break


if __name__ == "__main__":
    run_chatbot_with_human_in_loop()

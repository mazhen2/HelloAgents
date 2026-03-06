import os
from typing import Annotated
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages

# 设置环境变量
os.environ["DASHSCOPE_API_KEY"] = "sk-52254e84633242c9ae4383c7716486f3"
os.environ["DASHSCOPE_API_BASE"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# 导入 ChatQwen（在设置环境变量之后）
from langchain_qwq import ChatQwen


class State(TypedDict):
    messages: Annotated[list, add_messages]


graph_builder = StateGraph(State)

llm = ChatQwen(
    model="qwen-plus",
    temperature=0.7,
    timeout=60,
)


def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}


# The first argument is the unique node name
# The second argument is the function or object that will be called whenever
# the node is used.
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph = graph_builder.compile()

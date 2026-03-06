from pathlib import Path
from dotenv import load_dotenv
import os
from typing import Annotated

from langchain_qwq import ChatQwen
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages


def load_llm_config(env_path: str | Path = None) -> dict:
    """
    从 .env 文件加载 LLM 配置

    Args:
        env_path: .env 文件路径，默认为项目根目录

    Returns:
        包含配置的字典：model_name, api_key, base_url, timeout
    """
    if env_path is None:
        env_path = Path(__file__).parent.parent / ".env"
    elif isinstance(env_path, str):
        env_path = Path(env_path)

    # 加载 .env 文件
    load_dotenv(env_path)

    # 从环境变量读取配置
    model_name = os.getenv("LLM_MODEL_ID", "qwen-plus")
    api_key = os.getenv("LLM_API_KEY", "")
    base_url = os.getenv("LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    timeout = int(os.getenv("LLM_TIMEOUT", "60"))

    # 验证必要配置
    if not api_key:
        raise ValueError("未找到 LLM_API_KEY，请检查 .env 文件")

    # 设置到环境变量（供 ChatQwen 使用）
    os.environ["DASHSCOPE_API_KEY"] = api_key
    os.environ["DASHSCOPE_API_BASE"] = base_url

    # 打印配置信息
    print(f"✓ 模型：{model_name}")
    print(f"✓ API Key 已加载 (长度：{len(api_key)})")
    print(f"✓ Base URL: {base_url}")
    print(f"✓ Timeout: {timeout}秒")

    return {
        "model_name": model_name,
        "api_key": api_key,
        "base_url": base_url,
        "timeout": timeout
    }


class State(TypedDict):
    messages: Annotated[list, add_messages]


graph_builder = StateGraph(State)

# 加载配置并初始化模型
config = load_llm_config()

llm = ChatQwen(
    model=config["model_name"],
    temperature=0.7,
    timeout=config["timeout"],
)


def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}


graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph = graph_builder.compile()

# 运行聊天
result = graph.invoke({"messages": [("human", "你好")]})
print("\n" + "=" * 50)
print("🤖 AI 回复:")
print("=" * 50)
print(result["messages"][-1].content)
print("=" * 50 + "\n")
